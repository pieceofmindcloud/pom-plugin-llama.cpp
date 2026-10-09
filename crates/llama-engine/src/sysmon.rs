//! Host RAM / VRAM sampling, per platform.
//!
//! The two machines this runs on are not comparable, and the readings say so
//! instead of pretending otherwise:
//!
//! * **Linux + NVIDIA** — RAM from `/proc/meminfo`, VRAM from `nvidia-smi`,
//!   including the per-process breakdown so a number can be cross-checked
//!   against `nvidia-smi` by hand.
//! * **macOS / Apple Silicon** — RAM from `sysctl` + `vm_stat`. There is no
//!   separate VRAM pool: GPU buffers are wired pages of the same RAM. So the
//!   GPU section reports `kind: "unified"` with the wired total, and no
//!   `vram_total` at all, rather than inventing a dedicated-memory figure.
//! * **Windows** - RAM e disco vêm direto do kernel32 (`GlobalMemoryStatusEx`,
//!   `GetDiskFreeSpaceExW`), sem depender de nenhum executável externo. VRAM
//!   continua saindo do `nvidia-smi`, que o driver da NVIDIA instala no PATH;
//!   sem placa NVIDIA a seção de GPU volta vazia com um aviso, em vez de
//!   inventar número. Windows não tem load average: os três valores ficam em
//!   zero e a tela mostra isso como "não se aplica".

use serde::Serialize;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Chamadas ao kernel32 declaradas à mão.
///
/// São duas funções estáveis desde o Windows 2000 e ambas de assinatura
/// pequena, então declará-las aqui evita arrastar uma dependência inteira
/// (`windows-sys`) que só compila num alvo e cujo nome de *feature* não teria
/// como ser conferido a partir do macOS.
#[cfg(windows)]
mod win {
    /// Layout de `MEMORYSTATUSEX` (winbase.h). Os dois `u32` iniciais somam 8
    /// bytes, então os `u64` já caem alinhados: `repr(C)` reproduz o struct
    /// exatamente.
    /// A maior parte dos campos não é lida por nós, mas todos precisam existir:
    /// o kernel escreve o struct inteiro a partir de `dw_length`.
    #[repr(C)]
    #[allow(dead_code)]
    pub struct MemoryStatusEx {
        pub dw_length: u32,
        pub dw_memory_load: u32,
        pub ull_total_phys: u64,
        pub ull_avail_phys: u64,
        pub ull_total_page_file: u64,
        pub ull_avail_page_file: u64,
        pub ull_total_virtual: u64,
        pub ull_avail_virtual: u64,
        pub ull_avail_extended_virtual: u64,
    }

    /// `GetDriveTypeW` return value for a network share (winbase.h).
    pub const DRIVE_REMOTE: u32 = 4;

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        pub fn GetDiskFreeSpaceExW(
            directory_name: *const u16,
            free_bytes_available_to_caller: *mut u64,
            total_number_of_bytes: *mut u64,
            total_number_of_free_bytes: *mut u64,
        ) -> i32;
        pub fn GetDriveTypeW(root_path_name: *const u16) -> u32;
    }

    /// Um caminho como o Win32 espera: UTF-16 terminado em zero.
    pub fn wide(path: &std::path::Path) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    /// UNC (`\\server\share\...`) é sempre rede; uma letra de unidade também
    /// pode ser rede quando é uma unidade mapeada (`net use Z: \\server\share`),
    /// o que `GetDriveTypeW` é quem sabe dizer - o texto do caminho não distingue
    /// uma unidade mapeada de uma unidade local.
    pub fn is_network_path(path: &std::path::Path) -> bool {
        let text = path.to_string_lossy();
        if text.starts_with(r"\\") || text.starts_with("//") {
            return true;
        }
        let Some(root) = path.ancestors().last() else {
            return false;
        };
        // O componente raiz do Windows já vem como `C:\`, terminado em barra -
        // exatamente o que `GetDriveTypeW` espera.
        let root_wide = wide(root);
        unsafe { GetDriveTypeW(root_wide.as_ptr()) == DRIVE_REMOTE }
    }

    /// A unidade local do sistema (`%SystemDrive%`, tipicamente `C:\`), usada
    /// como âncora do disco quando o caminho pedido é de rede.
    pub fn local_system_drive() -> std::path::PathBuf {
        std::env::var("SystemDrive")
            .map(|drive| std::path::PathBuf::from(format!("{drive}\\")))
            .unwrap_or_else(|_| std::path::PathBuf::from(r"C:\"))
    }
}

/// Samples are cached this long; the UI polls faster than the numbers move,
/// and `nvidia-smi` costs a process spawn.
const CACHE_TTL: Duration = Duration::from_millis(1200);

/// Execute a short host utility without registering it with Tokio's SIGCHLD
/// reaper. Tokio's process reaper installs a process-wide signal handler, and
/// that handler can interrupt the blocking UDP receive used by the pkarr DHT.
/// The caller remains asynchronous because the std process is isolated in a
/// blocking worker.
async fn command_output(program: &'static str, args: Vec<String>) -> Option<std::process::Output> {
    tokio::task::spawn_blocking(move || std::process::Command::new(program).args(args).output())
        .await
        .ok()?
        .ok()
}

#[derive(Debug, Clone, Serialize)]
pub struct MemStat {
    pub total_mb: f64,
    pub used_mb: f64,
    pub available_mb: f64,
    pub used_pct: f64,
}

impl MemStat {
    fn new(total_mb: f64, available_mb: f64) -> Self {
        let used_mb = (total_mb - available_mb).max(0.0);
        Self {
            total_mb,
            used_mb,
            available_mb,
            used_pct: if total_mb > 0.0 {
                used_mb / total_mb * 100.0
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuStat {
    pub index: u32,
    pub name: String,
    /// De quem é esta placa. `gpu_kind` resume a máquina; isto identifica a
    /// peça - é o que permite ao painel mostrar uma Radeon e uma NVIDIA lado a
    /// lado sem adivinhar pelo nome.
    pub vendor: GpuVendor,
    /// Dedicated VRAM. `None` on unified-memory machines, where the concept
    /// does not apply.
    pub vram_total_mb: Option<f64>,
    pub vram_used_mb: Option<f64>,
    pub vram_used_pct: Option<f64>,
    /// GPU core utilisation, when the platform reports it.
    pub utilization_pct: Option<f64>,
    /// Board power draw, when the platform reports it.
    pub power_watts: Option<f64>,
    pub temperature_c: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    pub used_mb: f64,
}

/// O fabricante de uma placa específica.
///
/// Separado de `GpuKind` de propósito: `GpuKind` descreve a **máquina** (e por
/// isso tem `None` e `Mixed`), enquanto isto descreve **uma placa**, onde essas
/// duas variantes não fariam sentido.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Nvidia,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Amd,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Apple,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuKind {
    /// Discrete NVIDIA card with its own VRAM.
    Nvidia,
    /// Discrete AMD card with its own VRAM, servida pelo backend Vulkan.
    ///
    /// Mesma observação da `Unified`: só o Linux constrói esta variante, mas ela
    /// não pode sair por `cfg` porque `GpuKind` é serializado e um nó macOS
    /// mostra no painel o estado de peers que rodam com Radeon.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Amd,
    /// Placas de fabricantes diferentes na mesma máquina (NVIDIA + Radeon).
    ///
    /// Existe porque `gpu_kind` é um resumo, e resumir duas placas de marcas
    /// diferentes como uma delas esconderia a outra de quem lê só este campo.
    /// Quem precisa do detalhe lê `gpus[].vendor`.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Mixed,
    /// Apple Silicon: GPU and CPU share one memory pool.
    ///
    /// Só `read_gpu` no macOS constrói esta variante, então fora do macOS ela é
    /// dead code de verdade - e some do binário. Ela não pode ser cfg'ada para
    /// fora: `GpuKind` é o que o `/api/ui/*` serializa, e um nó Linux mostra no
    /// painel o estado de peers que rodam no MacBook.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Unified,
    /// Nothing detected.
    None,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemLoad {
    pub hostname: String,
    pub os: &'static str,
    pub arch: &'static str,
    pub cpu_count: usize,
    /// 1 / 5 / 15 minute load averages.
    pub load_avg: [f64; 3],
    pub ram: MemStat,
    pub gpu_kind: GpuKind,
    /// Plain-language caveat for whatever the GPU numbers mean here.
    pub gpu_note: Option<String>,
    pub gpus: Vec<GpuStat>,
    /// Processes holding GPU memory (NVIDIA only).
    pub gpu_processes: Vec<GpuProcess>,
    /// Memory wired down by the kernel — where Metal buffers live on macOS.
    pub wired_mb: Option<f64>,
    /// Free space where the model slices live, so the screen can say whether
    /// another slice fits before anyone starts downloading one.
    pub disk: DiskStat,
    /// Todo sistema de arquivos montado nesta máquina, um por partição.
    ///
    /// `disk` responde "cabe outra fatia?", que é uma pergunta sobre **um**
    /// volume - o do projeto. Isto responde outra: "que discos esta máquina
    /// tem?". Uma máquina com NVMe e HDD, ou com `/` e `/dados` separados,
    /// aparecia como um volume só, e quem decide onde pôr um GGUF de 40 GB não
    /// via o disco onde ele caberia.
    pub disks: Vec<DiskStat>,
    pub sampled_at: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DiskStat {
    pub total_mb: u64,
    pub free_mb: u64,
    pub mount_point: String,
    /// O dispositivo por trás do ponto de montagem (`/dev/nvme0n1p2`), quando a
    /// plataforma o informa. É o que distingue dois discos de duas partições do
    /// mesmo disco; vazio quando não se sabe.
    #[serde(default)]
    pub device: String,
}

// ── Disk ─────────────────────────────────────────────────────────────────────

/// Raiz do projeto onde as fatias moram, injetada por `crates/node` no arranque
/// (`sysmon::set_disk_root(config::project_root())`). `sysmon` não conhece o
/// layout do projeto; sem valor, mede o diretório de trabalho.
static DISK_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Fixa a raiz que `sample_disk()` mede. Uma segunda chamada é ignorada.
pub fn set_disk_root(root: PathBuf) {
    let _ = DISK_ROOT.set(root);
}

pub fn sample_disk() -> DiskStat {
    // The slices live under the project root, which is not necessarily the
    // working directory.
    match DISK_ROOT.get() {
        Some(root) => sample_disk_at(root),
        None => sample_disk_at(&std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
    }
}

/// Measure the volume that contains `path`, including when it is a configured
/// destination on another disk. A not-yet-created destination is probed via
/// its nearest existing parent, so planning never has to create directories.
pub fn sample_disk_at(path: &Path) -> DiskStat {
    let probe = path
        .ancestors()
        .find(|candidate| candidate.exists())
        .unwrap_or(path);
    let mount = path.to_string_lossy().to_string();

    // Windows não tem `df`; o volume que contém o diretório responde direto.
    #[cfg(windows)]
    {
        // O executável pode rodar a partir de uma pasta de rede compartilhada
        // (caminho UNC ou uma letra de unidade mapeada), sem que isso seja uma
        // cópia local. Medir esse volume relataria o disco da máquina que
        // hospeda o compartilhamento, não o disco local da máquina rodando o
        // processo - e todo peer apontando para a mesma pasta reportaria os
        // mesmos números. Quando o caminho é de rede, a métrica volta para a
        // unidade do sistema local (`%SystemDrive%`, tipicamente `C:`).
        let probe = if win::is_network_path(probe) {
            win::local_system_drive()
        } else {
            probe.to_path_buf()
        };
        let probe = probe.as_path();
        let mount = if win::is_network_path(path) {
            probe.to_string_lossy().to_string()
        } else {
            mount
        };
        let path = win::wide(probe);
        let (mut free_caller, mut total, mut free_total) = (0u64, 0u64, 0u64);
        let ok = unsafe {
            win::GetDiskFreeSpaceExW(path.as_ptr(), &mut free_caller, &mut total, &mut free_total)
        } != 0;
        if ok {
            // `free_caller` já desconta a cota do usuário, que é o número que
            // importa para saber se uma fatia cabe.
            return DiskStat {
                total_mb: total / 1_048_576,
                free_mb: free_caller / 1_048_576,
                mount_point: mount,
                device: String::new(),
            };
        }
        return DiskStat {
            total_mb: 0,
            free_mb: 0,
            mount_point: mount,
            device: String::new(),
        };
    }

    #[cfg(not(windows))]
    {
        let out = std::process::Command::new("df")
            .arg("-m")
            .arg(probe)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok());

        if let Some(output) = out {
            let lines: Vec<&str> = output.lines().collect();
            if lines.len() >= 2 {
                let fields: Vec<&str> = lines[1].split_whitespace().collect();
                if fields.len() >= 4 {
                    if let (Ok(total), Ok(free)) =
                        (fields[1].parse::<u64>(), fields[3].parse::<u64>())
                    {
                        return DiskStat {
                            total_mb: total,
                            free_mb: free,
                            mount_point: mount,
                            device: fields[0].to_string(),
                        };
                    }
                }
            }
        }

        DiskStat {
            total_mb: 0,
            free_mb: 0,
            mount_point: mount,
            device: String::new(),
        }
    }
}

/// Nomes que o `df` usa para coisas que não são disco.
///
/// Um `tmpfs` é RAM e um `devfs` não guarda byte nenhum; listá-los ao lado do
/// NVMe faria a tela prometer espaço que não existe. O corte é por nome do
/// dispositivo porque o `df -P` não informa o tipo do sistema de arquivos.
const PSEUDO_FILESYSTEMS: &[&str] = &[
    "devfs", "devtmpfs", "tmpfs", "overlay", "squashfs", "efivarfs", "udev", "none", "map", "shm",
];

/// Montagens de infraestrutura não são discos úteis para quem opera o POM.
///
/// A lista é aplicada à saída comum de `df` para Linux e macOS. Volumes de
/// dados continuam visíveis, incluindo `/System/Volumes/Data`; no Windows a
/// enumeração segue limitada ao volume do projeto pelo caminho nativo.
fn is_user_data_mount(mount_point: &str) -> bool {
    let mount = mount_point.trim_end_matches('/').to_ascii_lowercase();
    if mount == "/boot" || mount == "/efi" || mount == "/boot/efi" {
        return false;
    }
    if mount.starts_with("/boot/efi/") || mount.starts_with("/efi/") {
        return false;
    }
    if let Some(suffix) = mount.strip_prefix("/system/volumes/") {
        return suffix == "data";
    }
    if let Some(suffix) = mount.strip_prefix("/volumes/") {
        let volume = suffix.split('/').next().unwrap_or_default();
        return !matches!(
            volume,
            "preboot" | "vm" | "recovery" | "update" | "hardware" | "iscpreboot" | "xarts"
        );
    }
    true
}

/// Uma linha do `df -k -P`, lida da direita para a esquerda.
///
/// Da direita porque só o **fim** da linha tem posição garantida: o POSIX fixa
/// seis colunas, mas o nome do dispositivo pode conter espaço (`map auto_home`
/// no macOS) e empurraria tudo se a leitura começasse pela esquerda. Ponto de
/// montagem com espaço continua fora do alcance de qualquer análise por
/// colunas, e o `df` não oferece outra saída.
///
/// A unidade é KiB, não MB: `-P` fixa blocos de 512 bytes no macOS e ignora o
/// `-m`, então pedir megabytes ali devolvia números 2048 vezes maiores. `-k` é
/// a unidade que o POSIX garante nos dois sistemas.
fn parse_df_line(line: &str) -> Option<DiskStat> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 6 {
        return None;
    }
    let last = fields.len() - 1;
    let mount_point = fields[last].to_string();
    let total_mb = fields[last - 4].parse::<u64>().ok()? / 1024;
    let free_mb = fields[last - 2].parse::<u64>().ok()? / 1024;
    let device = fields[..last - 4].join(" ");
    Some(DiskStat {
        total_mb,
        free_mb,
        mount_point,
        device,
    })
}

/// Os volumes reais de uma saída de `df -k -P`, sem os pseudo sistemas de
/// arquivos e sem repetir um ponto de montagem.
fn parse_df_all(output: &str) -> Vec<DiskStat> {
    let mut out: Vec<DiskStat> = Vec::new();
    for line in output.lines().skip(1) {
        let Some(disk) = parse_df_line(line) else {
            continue;
        };
        // Volume de tamanho zero não é disco: é uma montagem sintética.
        if disk.total_mb == 0 {
            continue;
        }
        let head = disk.device.split_whitespace().next().unwrap_or_default();
        if PSEUDO_FILESYSTEMS.contains(&head) {
            continue;
        }
        if !is_user_data_mount(&disk.mount_point) {
            continue;
        }
        if out.iter().any(|seen| seen.mount_point == disk.mount_point) {
            continue;
        }
        out.push(disk);
    }
    out.sort_by(|a, b| a.mount_point.cmp(&b.mount_point));
    out
}

/// Todo volume montado nesta máquina.
///
/// Lista vazia quer dizer "não consegui perguntar", e a tela mostra isso como
/// desconhecido em vez de "esta máquina não tem disco".
pub fn sample_all_disks() -> Vec<DiskStat> {
    // Windows não tem `df`; sem uma enumeração de volumes escrita à mão, o
    // volume do projeto continua sendo o único que esta plataforma reporta.
    #[cfg(windows)]
    {
        let only = sample_disk();
        return if only.total_mb > 0 {
            vec![only]
        } else {
            Vec::new()
        };
    }

    #[cfg(not(windows))]
    {
        let out = std::process::Command::new("df")
            .args(["-k", "-P"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok());
        out.as_deref().map(parse_df_all).unwrap_or_default()
    }
}

// ── Cache ────────────────────────────────────────────────────────────────────

struct Cached {
    at: Instant,
    load: SystemLoad,
}

fn cache() -> &'static Mutex<Option<Cached>> {
    static CACHE: OnceLock<Mutex<Option<Cached>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Current host load, cached for [`CACHE_TTL`].
pub async fn sample() -> SystemLoad {
    {
        let guard = cache().lock().await;
        if let Some(ref c) = *guard {
            if c.at.elapsed() < CACHE_TTL {
                return c.load.clone();
            }
        }
    }

    let load = sample_uncached().await;
    *cache().lock().await = Some(Cached {
        at: Instant::now(),
        load: load.clone(),
    });
    load
}

async fn sample_uncached() -> SystemLoad {
    let (ram, wired_mb) = read_memory().await;
    let (gpu_kind, gpu_note, gpus, gpu_processes) = read_gpu(&ram).await;

    SystemLoad {
        hostname: hostname().await,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpu_count: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        load_avg: read_load_avg().await,
        disk: tokio::task::spawn_blocking(sample_disk)
            .await
            .unwrap_or_else(|_| DiskStat::default()),
        disks: tokio::task::spawn_blocking(sample_all_disks)
            .await
            .unwrap_or_default(),
        ram,
        gpu_kind,
        gpu_note,
        gpus,
        gpu_processes,
        wired_mb,
        sampled_at: chrono::Utc::now().to_rfc3339(),
    }
}

// ── Memory ───────────────────────────────────────────────────────────────────

/// Host RAM alone, without the GPU and disk probes of the full sample.
pub async fn sample_memory() -> MemStat {
    read_memory().await.0
}

/// Returns (RAM, wired pages in MB).
async fn read_memory() -> (MemStat, Option<f64>) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(txt) = tokio::fs::read_to_string("/proc/meminfo").await {
            let kb = |key: &str| -> Option<f64> {
                txt.lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<f64>().ok())
            };
            let total = kb("MemTotal:").unwrap_or(0.0) / 1024.0;
            // MemAvailable is the kernel's own estimate of what a new workload
            // could get — closer to the truth than MemFree.
            let avail = kb("MemAvailable:").unwrap_or(0.0) / 1024.0;
            return (MemStat::new(total, avail), None);
        }
    }

    #[cfg(target_os = "macos")]
    {
        let total_bytes = sysctl_number("hw.memsize").await.unwrap_or(0.0);
        let total_mb = total_bytes / 1_048_576.0;
        if let Some((used_mb, wired_mb)) = vm_stat_used().await {
            return (
                MemStat::new(total_mb, (total_mb - used_mb).max(0.0)),
                Some(wired_mb),
            );
        }
        return (MemStat::new(total_mb, 0.0), None);
    }

    #[cfg(windows)]
    {
        // `ullAvailPhys` é o equivalente honesto do MemAvailable do Linux: o
        // que uma carga nova conseguiria sem paginar.
        let mut st = win::MemoryStatusEx {
            dw_length: std::mem::size_of::<win::MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        let ok = unsafe { win::GlobalMemoryStatusEx(&mut st) } != 0;
        if ok {
            let mb = |b: u64| b as f64 / 1_048_576.0;
            return (
                MemStat::new(mb(st.ull_total_phys), mb(st.ull_avail_phys)),
                None,
            );
        }
        // Falhou: zeros dizem "não sei", e é assim que o resto do sistema lê
        // (ver os guardas `> 0` em `deploy.rs`).
        return (MemStat::new(0.0, 0.0), None);
    }

    #[allow(unreachable_code)]
    (MemStat::new(0.0, 0.0), None)
}

#[cfg(target_os = "macos")]
async fn sysctl_number(key: &str) -> Option<f64> {
    let out = command_output("sysctl", vec!["-n".into(), key.to_string()]).await?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .ok()
}

/// Parse `vm_stat` into (used MB, wired MB), matching how Activity Monitor
/// counts "Memory Used": active + wired + compressed.
#[cfg(target_os = "macos")]
async fn vm_stat_used() -> Option<(f64, f64)> {
    let out = command_output("vm_stat", Vec::new()).await?;
    let txt = String::from_utf8_lossy(&out.stdout);

    let page_size = txt
        .lines()
        .next()
        .and_then(|l| {
            l.split("page size of ")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
        })
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(16384.0);

    let pages = |key: &str| -> f64 {
        txt.lines()
            .find(|l| l.trim_start().starts_with(key))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.trim().trim_end_matches('.'))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
    };

    let active = pages("Pages active");
    let wired = pages("Pages wired down");
    let compressed = pages("Pages occupied by compressor");

    let to_mb = |p: f64| p * page_size / 1_048_576.0;
    Some((to_mb(active + wired + compressed), to_mb(wired)))
}

// ── GPU ──────────────────────────────────────────────────────────────────────

async fn read_gpu(ram: &MemStat) -> (GpuKind, Option<String>, Vec<GpuStat>, Vec<GpuProcess>) {
    // As duas famílias são somadas, não escolhidas: numa máquina com Radeon e
    // NVIDIA juntas, reportar só uma esconderia metade do hardware de quem
    // decide onde implantar. `gpu_kind` vira o resumo; cada placa carrega o
    // próprio `vendor`.
    // Só o ramo Linux abaixo estende a lista, então fora dele o `mut` sobra -
    // e tirá-lo por `cfg` partiria a linha em duas versões do mesmo código.
    #[cfg(target_os = "linux")]
    let pci_ids = read_pci_ids().await;

    let backend_gpus = backend_gpu_devices();
    let (mut gpus, procs) = nvidia_smi().await.unwrap_or_default();

    #[cfg(target_os = "linux")]
    let mut procs = procs;
    #[cfg(target_os = "linux")]
    {
        // Índices continuam a contagem da NVIDIA para não haver dois `index: 0`
        // - a interface usa esse número para falar de uma placa específica.
        let amd = amd_sysfs(gpus.len() as u32, &backend_gpus, pci_ids.as_deref()).await;
        if !amd.is_empty() {
            procs.extend(amd_gpu_processes().await);
        }
        gpus.extend(amd);
    }

    // O nó só reporta as placas que pode usar. Os índices originais são
    // preservados de propósito: é por eles que a `pom.json` e o painel falam de
    // uma placa, e renumerar faria `gpus: [1]` aparecer como `#0`.
    if let Ok(choice) = SELECTION.read() {
        if let Some(allowed) = choice.as_ref() {
            gpus.retain(|g| allowed.contains(&g.index));
        }
    }
    let nvidia = gpus
        .iter()
        .filter(|g| g.vendor == GpuVendor::Nvidia)
        .count();
    let amd = gpus.iter().filter(|g| g.vendor == GpuVendor::Amd).count();

    if !gpus.is_empty() {
        let kind = match (nvidia, amd) {
            (0, _) => GpuKind::Amd,
            (_, 0) => GpuKind::Nvidia,
            _ => GpuKind::Mixed,
        };
        let note = (amd > 0).then(|| {
            "Radeon lida pelo sysfs do amdgpu e servida pelo backend Vulkan \
             (compile com `--features vulkan`). Os processos com VRAM em mão \
             saem de /proc/<pid>/fdinfo, então a lista cobre o que este \
             usuário pode ler - o próprio nó sempre entra."
                .to_string()
        });
        return (kind, note, gpus, procs);
    }

    #[cfg(target_os = "macos")]
    {
        if std::env::consts::ARCH == "aarch64" {
            let name = mac_gpu_name(&backend_gpus)
                .await
                .unwrap_or_else(|| "Apple Silicon GPU".to_string());
            // O acelerador publica as próprias estatísticas no IORegistry, sem
            // root e sem `powermetrics`: uso do núcleo e memória de sistema
            // que a GPU tem em mão. É a única leitura de GPU que este alvo
            // oferece a um processo comum - a temperatura mora no SMC, atrás
            // de API privada, e por isso continua desconhecida.
            let stats = mac_accelerator_stats().await.unwrap_or_default();
            return (
                GpuKind::Unified,
                Some(
                    "Memória unificada: a GPU usa a mesma RAM do sistema, \
                     não existe VRAM dedicada para medir. O que aparece como \
                     memória da GPU é a memória de sistema que o acelerador \
                     tem em uso, lida do IORegistry; a temperatura fica no \
                     SMC, que não é legível sem privilégio."
                        .to_string(),
                ),
                vec![GpuStat {
                    index: 0,
                    name,
                    vendor: GpuVendor::Apple,
                    vram_total_mb: None,
                    vram_used_mb: stats.in_use_memory_mb,
                    vram_used_pct: None,
                    utilization_pct: stats.utilization_pct,
                    power_watts: None,
                    temperature_c: None,
                }],
                Vec::new(),
            );
        }
    }

    #[cfg(windows)]
    {
        // Sem NVIDIA não há de onde ler VRAM sem falar com DXGI ou WMI, e
        // nenhum dos dois diria quanto da placa está em uso pelo nosso
        // processo. Então o honesto é não reportar GPU nenhuma e dizer por quê.
        return (
            GpuKind::None,
            Some(
                "Nenhuma GPU NVIDIA encontrada (o `nvidia-smi` não respondeu). \
                 Neste alvo a inferência roda em CPU e não há leitura de VRAM \
                 para mostrar."
                    .to_string(),
            ),
            Vec::new(),
            Vec::new(),
        );
    }

    #[allow(unreachable_code)]
    {
        let _ = ram;
        (GpuKind::None, None, Vec::new(), Vec::new())
    }
}

/// Lê a VRAM usada pelo processo atual, somando as linhas do próprio PID.
///
/// Isto é deliberadamente uma sondagem pontual e síncrona: o caminho só é
/// usado durante o fit, onde precisamos correlacionar o valor do driver com o
/// snapshot que o llama.cpp recebeu. Em máquinas sem NVIDIA, ou quando o
/// `nvidia-smi` não está disponível, devolve `None`.
pub fn current_process_gpu_memory_mb() -> Option<f64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    parse_current_process_gpu_memory_mb(
        &String::from_utf8_lossy(&output.stdout),
        std::process::id(),
    )
}

fn parse_current_process_gpu_memory_mb(output: &str, pid: u32) -> Option<f64> {
    let mut total_mb = 0.0;
    let mut found = false;
    for line in output.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 2 || fields[0].parse::<u32>().ok() != Some(pid) {
            continue;
        }
        let used_mb = fields[1].parse::<f64>().ok()?;
        total_mb += used_mb;
        found = true;
    }
    found.then_some(total_mb)
}

/// Query NVIDIA cards. Returns `None` when `nvidia-smi` is absent or fails,
/// which is the normal case on the MacBook.
async fn nvidia_smi() -> Option<(Vec<GpuStat>, Vec<GpuProcess>)> {
    let out = command_output(
        "nvidia-smi",
        vec![
            "--query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw".into(),
            "--format=csv,noheader,nounits".into(),
        ],
    )
    .await?;
    if !out.status.success() {
        return None;
    }

    let mut gpus = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if f.len() < 8 {
            continue;
        }
        let total: f64 = f[3].parse().unwrap_or(0.0);
        let used: f64 = f[4].parse().unwrap_or(0.0);
        let fallback = format!("NVIDIA GPU ({})", f[2]);
        gpus.push(GpuStat {
            index: f[0].parse().unwrap_or(0),
            name: resolve_gpu_name(Some(f[1]), None, fallback),
            vendor: GpuVendor::Nvidia,
            vram_total_mb: Some(total),
            vram_used_mb: Some(used),
            vram_used_pct: if total > 0.0 {
                Some(used / total * 100.0)
            } else {
                None
            },
            utilization_pct: f[5].parse().ok(),
            power_watts: f[7].parse().ok(),
            temperature_c: f[6].parse().ok(),
        });
    }

    let mut procs = Vec::new();
    if let Some(out) = command_output(
        "nvidia-smi",
        vec![
            "--query-compute-apps=pid,process_name,used_memory".into(),
            "--format=csv,noheader,nounits".into(),
        ],
    )
    .await
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if f.len() < 3 {
                continue;
            }
            procs.push(GpuProcess {
                pid: f[0].parse().unwrap_or(0),
                name: f[1].rsplit('/').next().unwrap_or(f[1]).to_string(),
                used_mb: f[2].parse().unwrap_or(0.0),
            });
        }
    }

    Some((gpus, procs))
}

/// As Radeon desta máquina, lidas do sysfs do `amdgpu`.
///
/// Sem binário externo de propósito: o equivalente ao `nvidia-smi` seria o
/// `rocm-smi`, que só existe se o ROCm inteiro estiver instalado - e o backend
/// que usamos é o Vulkan, que não depende de ROCm. O `amdgpu` já publica tudo
/// o que o painel mostra, e uma máquina sem Radeon simplesmente não tem os
/// arquivos.
///
/// O driver proprietário da NVIDIA não usa este sysfs, então o filtro por
/// `vendor == 0x1002` nunca confunde as duas placas de uma máquina mista.
#[cfg(target_os = "linux")]
async fn amd_sysfs(
    first_index: u32,
    backend_gpus: &[BackendGpu],
    pci_ids: Option<&str>,
) -> Vec<GpuStat> {
    /// Vendor PCI da AMD/ATI.
    const AMD: &str = "0x1002";

    let read = |p: std::path::PathBuf| async move {
        tokio::fs::read_to_string(p)
            .await
            .ok()
            .map(|v| v.trim().to_string())
    };

    let mut entries = match tokio::fs::read_dir("/sys/class/drm").await {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    // `card0`, `card1`, ... sem os conectores (`card1-DP-2`), que apontam para o
    // mesmo dispositivo e contariam a placa várias vezes.
    let mut cards: Vec<String> = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("card") && !name[4..].contains('-') {
            cards.push(name);
        }
    }
    cards.sort();

    let mut out = Vec::new();
    for name in cards {
        let dev = std::path::Path::new("/sys/class/drm")
            .join(&name)
            .join("device");
        if read(dev.join("vendor")).await.as_deref() != Some(AMD) {
            continue;
        }

        let mb = |bytes: Option<String>| {
            bytes
                .and_then(|v| v.parse::<f64>().ok())
                .map(|b| b / 1024.0 / 1024.0)
        };
        let total = mb(read(dev.join("mem_info_vram_total")).await);
        let used = mb(read(dev.join("mem_info_vram_used")).await);

        let device_id = read(dev.join("device")).await;
        let pci_bus_id = read(dev.join("uevent"))
            .await
            .and_then(|uevent| uevent_value(&uevent, "PCI_SLOT_NAME"));
        let fallback = format!(
            "AMD Radeon ({})",
            device_id.as_deref().unwrap_or("unknown PCI device")
        );
        let pci_name = pci_ids.and_then(|ids| {
            pci_device_name(
                ids,
                AMD.trim_start_matches("0x"),
                device_id
                    .as_deref()
                    .unwrap_or_default()
                    .trim_start_matches("0x"),
            )
        });

        out.push(GpuStat {
            index: first_index + out.len() as u32,
            vendor: GpuVendor::Amd,
            name: resolve_gpu_name(
                backend_gpu_name(backend_gpus, "Vulkan", pci_bus_id.as_deref()),
                pci_name.as_deref(),
                fallback,
            ),
            vram_total_mb: total,
            vram_used_mb: used,
            vram_used_pct: match (total, used) {
                (Some(t), Some(u)) if t > 0.0 => Some(u / t * 100.0),
                _ => None,
            },
            utilization_pct: read(dev.join("gpu_busy_percent"))
                .await
                .and_then(|v| v.parse::<f64>().ok()),
            power_watts: amd_power_watts(&dev).await,
            temperature_c: amd_temperature(&dev).await,
        });
    }
    out
}

/// Um valor de `fdinfo` do DRM: número mais unidade (`1024 KiB`), em bytes.
///
/// O kernel escreve KiB na maioria dos campos, mas a especificação
/// (`drm-usage-stats`) permite outras unidades, e ler o número sem olhar a
/// unidade é como um "1 MiB" vira "1 byte".
#[cfg(any(target_os = "linux", test))]
fn parse_drm_size_bytes(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let amount = parts.next()?.parse::<u64>().ok()?;
    let unit = parts.next().unwrap_or("B");
    let factor = match unit {
        "B" => 1,
        "KiB" => 1024,
        "MiB" => 1024 * 1024,
        "GiB" => 1024 * 1024 * 1024,
        _ => return None,
    };
    Some(amount * factor)
}

/// O que um `fdinfo` de um descritor do amdgpu diz: cliente e VRAM em mão.
///
/// `None` quando o descritor não é do amdgpu - a maioria deles não é, e este é
/// o filtro que faz a varredura ser barata.
#[cfg(any(target_os = "linux", test))]
fn parse_amdgpu_fdinfo(text: &str) -> Option<(String, u64)> {
    let field = |key: &str| -> Option<&str> {
        text.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim() == key).then(|| value.trim())
        })
    };
    if field("drm-driver")? != "amdgpu" {
        return None;
    }
    let client = field("drm-client-id")?.to_string();
    // `drm-resident-vram` é o nome novo do mesmo número; onde os dois existem,
    // ele é o que conta as páginas de fato residentes.
    let vram = field("drm-resident-vram")
        .or_else(|| field("drm-memory-vram"))
        .and_then(parse_drm_size_bytes)
        .unwrap_or(0);
    Some((client, vram))
}

/// Quem segura VRAM nas Radeon desta máquina, lido de `/proc/<pid>/fdinfo`.
///
/// É o equivalente do que o `nvidia-smi` entrega de graça nas NVIDIA, e a
/// razão de a seção "processos na GPU" ficar vazia numa máquina AMD: nada a
/// preenchia. O kernel publica por descritor, então o mesmo contexto aparece
/// várias vezes - `drm-client-id` é o que desduplica.
///
/// Um processo de outro usuário recusa a leitura, e recusa é o caso comum:
/// esta lista é honestamente **o que este usuário pode ver**, não a máquina
/// inteira. O processo que interessa - o próprio nó - está sempre nela.
#[cfg(target_os = "linux")]
async fn amd_gpu_processes() -> Vec<GpuProcess> {
    let mut out: Vec<GpuProcess> = Vec::new();
    let Ok(mut procs) = tokio::fs::read_dir("/proc").await else {
        return out;
    };
    while let Ok(Some(entry)) = procs.next_entry().await {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(mut fds) = tokio::fs::read_dir(entry.path().join("fdinfo")).await else {
            continue;
        };
        let mut seen: Vec<String> = Vec::new();
        let mut bytes = 0u64;
        while let Ok(Some(fd)) = fds.next_entry().await {
            let Ok(text) = tokio::fs::read_to_string(fd.path()).await else {
                continue;
            };
            let Some((client, vram)) = parse_amdgpu_fdinfo(&text) else {
                continue;
            };
            if seen.contains(&client) {
                continue;
            }
            seen.push(client);
            bytes += vram;
        }
        if bytes == 0 {
            continue;
        }
        let comm = tokio::fs::read_to_string(entry.path().join("comm"))
            .await
            .map(|c| c.trim().to_string())
            .unwrap_or_default();
        out.push(GpuProcess {
            pid,
            name: if comm.is_empty() {
                format!("pid {pid}")
            } else {
                comm
            },
            used_mb: bytes as f64 / 1_048_576.0,
        });
    }
    out.sort_by(|a, b| b.used_mb.total_cmp(&a.used_mb));
    out
}

/// Temperatura da placa, em `hwmon/hwmon<N>/temp1_input` (milésimos de grau).
///
/// O `<N>` é atribuído na ordem em que os drivers registram, então varia entre
/// arranques e não pode ser fixado - daí a varredura do diretório.
#[cfg(target_os = "linux")]
async fn amd_temperature(device: &std::path::Path) -> Option<f64> {
    let mut hwmon = tokio::fs::read_dir(device.join("hwmon")).await.ok()?;
    while let Ok(Some(entry)) = hwmon.next_entry().await {
        if let Ok(raw) = tokio::fs::read_to_string(entry.path().join("temp1_input")).await {
            if let Ok(milli) = raw.trim().parse::<f64>() {
                return Some(milli / 1000.0);
            }
        }
    }
    None
}

/// Consumo da placa AMD no mesmo hwmon que publica a temperatura.
///
/// O driver costuma expor `power1_average` em microwatts, mas algumas versões
/// só oferecem `power1_input`. Se nenhuma existir, o painel mantém o consumo
/// desconhecido em vez de tratar uma leitura ausente como zero.
#[cfg(target_os = "linux")]
async fn amd_power_watts(device: &std::path::Path) -> Option<f64> {
    let mut hwmon = tokio::fs::read_dir(device.join("hwmon")).await.ok()?;
    while let Ok(Some(entry)) = hwmon.next_entry().await {
        for field in ["power1_average", "power1_input"] {
            if let Ok(raw) = tokio::fs::read_to_string(entry.path().join(field)).await {
                if let Ok(microwatts) = raw.trim().parse::<f64>() {
                    return Some(microwatts / 1_000_000.0);
                }
            }
        }
    }
    None
}

/// O que o acelerador da Apple publica sobre si mesmo.
///
/// Campo ausente é `None`, nunca zero: "a GPU está a 0%" e "não perguntei" são
/// respostas diferentes, e a tela mostra as duas de forma diferente.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct MacAcceleratorStats {
    utilization_pct: Option<f64>,
    in_use_memory_mb: Option<f64>,
}

/// Lê `PerformanceStatistics` do acelerador no IORegistry.
///
/// `AGXAccelerator` é a classe do Apple Silicon; `IOAccelerator` cobre os Macs
/// Intel, onde a GPU é discreta ou integrada da Intel/AMD. A saída de texto do
/// `ioreg` é usada de propósito em vez do plist (`-a`): o dicionário de
/// estatísticas cabe numa linha e a leitura vira uma varredura de chaves, sem
/// arrastar um parser de XML para dentro do monitor.
#[cfg(target_os = "macos")]
async fn mac_accelerator_stats() -> Option<MacAcceleratorStats> {
    for class in ["AGXAccelerator", "IOAccelerator"] {
        let out = command_output(
            "ioreg",
            vec![
                "-r".into(),
                "-d".into(),
                "1".into(),
                "-w".into(),
                "0".into(),
                "-c".into(),
                class.to_string(),
            ],
        )
        .await;
        let Some(out) = out else { continue };
        let stats = parse_ioreg_accelerator(&String::from_utf8_lossy(&out.stdout));
        if stats != MacAcceleratorStats::default() {
            return Some(stats);
        }
    }
    None
}

/// O valor numérico de uma chave dentro do dicionário do `ioreg`.
///
/// As chaves vêm entre aspas e o valor logo depois do `=`, com ou sem espaço:
/// `"Device Utilization %"=37`.
#[cfg(any(target_os = "macos", test))]
fn ioreg_number(text: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let at = text.find(&needle)? + needle.len();
    let rest = text[at..].trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let digits: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    digits.parse().ok()
}

#[cfg(any(target_os = "macos", test))]
fn parse_ioreg_accelerator(text: &str) -> MacAcceleratorStats {
    MacAcceleratorStats {
        utilization_pct: ioreg_number(text, "Device Utilization %"),
        // O IORegistry publica bytes; a tela fala em MB como no resto do painel.
        in_use_memory_mb: ioreg_number(text, "In use system memory")
            .map(|bytes| bytes / 1_048_576.0),
    }
}

/// Fora do macOS o tipo existe só para o teste do parser compilar.
#[cfg(all(not(target_os = "macos"), test))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct MacAcceleratorStats {
    utilization_pct: Option<f64>,
    in_use_memory_mb: Option<f64>,
}

#[cfg(target_os = "macos")]
async fn mac_gpu_name(backend_gpus: &[BackendGpu]) -> Option<String> {
    if let Some(name) = backend_gpu_name(backend_gpus, "Metal", None) {
        return Some(name.to_string());
    }

    let out = command_output("system_profiler", vec!["SPDisplaysDataType".into()]).await?;
    system_profiler_gpu_name(&String::from_utf8_lossy(&out.stdout))
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
struct BackendGpu {
    backend: String,
    description: String,
    device_id: Option<String>,
}

fn backend_gpu_devices() -> Vec<BackendGpu> {
    if !BACKEND_INITIALIZED.load(Ordering::Acquire) {
        return Vec::new();
    }

    unsafe {
        let mut raw = ptr::null_mut();
        let status = llama_cpp_sys_2::llama_rs_backend_devices_json(&mut raw);
        if status != llama_cpp_sys_2::LLAMA_RS_STATUS_OK || raw.is_null() {
            return Vec::new();
        }
        let text = CStr::from_ptr(raw).to_string_lossy().into_owned();
        llama_cpp_sys_2::llama_rs_string_free(raw);
        parse_backend_gpu_devices(&text)
    }
}

fn parse_backend_gpu_devices(text: &str) -> Vec<BackendGpu> {
    serde_json::from_str(text).unwrap_or_default()
}

static BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub fn mark_backend_initialized() {
    BACKEND_INITIALIZED.store(true, Ordering::Release);
}

fn backend_gpu_name<'a>(
    entries: &'a [BackendGpu],
    backend: &str,
    device_id: Option<&str>,
) -> Option<&'a str> {
    entries
        .iter()
        .filter(|entry| entry.backend.eq_ignore_ascii_case(backend))
        .find(|entry| {
            let same_device = match device_id {
                Some(id) => entry
                    .device_id
                    .as_deref()
                    .is_some_and(|candidate| candidate.eq_ignore_ascii_case(id)),
                None => true,
            };
            same_device && !entry.description.trim().is_empty()
        })
        .map(|entry| entry.description.trim())
}

fn resolve_gpu_name(primary: Option<&str>, secondary: Option<&str>, fallback: String) -> String {
    primary
        .into_iter()
        .chain(secondary)
        .map(str::trim)
        .find(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            if fallback.trim().is_empty() {
                "GPU desconhecida".to_string()
            } else {
                fallback
            }
        })
}

#[cfg(target_os = "linux")]
fn uevent_value(uevent: &str, key: &str) -> Option<String> {
    uevent.lines().find_map(|line| {
        let (candidate, value) = line.split_once('=')?;
        (candidate == key).then(|| value.trim().to_string())
    })
}

#[cfg(target_os = "linux")]
async fn read_pci_ids() -> Option<String> {
    for path in ["/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids"] {
        if let Ok(ids) = tokio::fs::read_to_string(path).await {
            return Some(ids);
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn pci_device_name(contents: &str, vendor: &str, device: &str) -> Option<String> {
    let vendor = vendor.trim_start_matches("0x");
    let device = device.trim_start_matches("0x");
    let mut in_vendor = false;
    for line in contents.lines() {
        if !line.starts_with('\t') {
            in_vendor = line.split_whitespace().next() == Some(vendor);
            continue;
        }
        if in_vendor && !line.starts_with("\t\t") {
            let mut fields = line.trim_start_matches('\t').splitn(2, char::is_whitespace);
            if fields
                .next()
                .is_some_and(|id| id.eq_ignore_ascii_case(device))
            {
                let name = fields.next()?.trim();
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

#[cfg(any(target_os = "macos", test))]
fn system_profiler_gpu_name(contents: &str) -> Option<String> {
    contents
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.trim() == "Chipset Model")
        .map(|(_, value)| value.trim().to_string())
        .filter(|name| !name.is_empty())
}

// ── Misc host facts ──────────────────────────────────────────────────────────

/// Load average de 1/5/15 minutos.
///
/// Windows não tem o conceito: o contador equivalente do WMI mede fila de
/// processador, que não é a mesma grandeza e enganaria quem compara os nós
/// lado a lado. Lá os três valores ficam em zero de propósito.
async fn read_load_avg() -> [f64; 3] {
    #[cfg(target_os = "linux")]
    {
        if let Ok(txt) = tokio::fs::read_to_string("/proc/loadavg").await {
            let v: Vec<f64> = txt
                .split_whitespace()
                .take(3)
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() == 3 {
                return [v[0], v[1], v[2]];
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(out) = command_output("sysctl", vec!["-n".into(), "vm.loadavg".into()]).await {
            let txt = String::from_utf8_lossy(&out.stdout);
            let v: Vec<f64> = txt
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 3 {
                return [v[0], v[1], v[2]];
            }
        }
    }

    [0.0, 0.0, 0.0]
}

async fn hostname() -> String {
    static CACHED: OnceLock<String> = OnceLock::new();
    if let Some(h) = CACHED.get() {
        return h.clone();
    }
    // No Windows o nome vem de COMPUTERNAME, que o sistema sempre define, e
    // isso poupa um spawn de processo. O `hostname.exe` existe e responderia,
    // mas cada spawn abre um console por um instante.
    #[cfg(windows)]
    let name = std::env::var("COMPUTERNAME").unwrap_or_default();

    #[cfg(not(windows))]
    let name = command_output("hostname", Vec::new())
        .await
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_default();

    let name = if name.trim().is_empty() {
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string())
    } else {
        name.trim().to_string()
    };
    let _ = CACHED.set(name.clone());
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_stat_derives_used_and_pct() {
        let m = MemStat::new(1000.0, 250.0);
        assert_eq!(m.used_mb, 750.0);
        assert!((m.used_pct - 75.0).abs() < 1e-9);
    }

    #[test]
    fn mem_stat_never_reports_negative_use() {
        let m = MemStat::new(100.0, 200.0);
        assert_eq!(m.used_mb, 0.0);
    }

    #[tokio::test]
    async fn sample_reports_real_ram() {
        let s = sample().await;
        assert!(s.ram.total_mb > 0.0, "total RAM should be discoverable");
        assert!(s.cpu_count > 0);
    }

    /// A leitura de verdade desta máquina, conferida contra a outra chamada de
    /// `df` que o nó já fazia. É este cruzamento que pega uma troca de unidade:
    /// `-P` fixa blocos de 512 bytes no macOS e ignora o `-m`, e sem esta
    /// conferência os volumes apareceriam 2048 vezes maiores do que são.
    #[cfg(not(windows))]
    #[tokio::test]
    async fn this_machine_volumes_match_the_project_df() {
        let disks = tokio::task::spawn_blocking(sample_all_disks).await.unwrap();
        assert!(
            !disks.is_empty(),
            "a máquina deveria ter ao menos um volume"
        );
        for disk in &disks {
            assert!(disk.total_mb > 0, "{} sem tamanho", disk.mount_point);
            assert!(
                disk.free_mb <= disk.total_mb,
                "{} livre > total",
                disk.mount_point
            );
        }

        let root = disks
            .iter()
            .find(|d| d.mount_point == "/")
            .expect("a raiz deveria estar montada");
        let reference = sample_disk_at(Path::new("/"));
        let difference = root.total_mb.abs_diff(reference.total_mb) as f64;
        assert!(
            difference / (reference.total_mb.max(1) as f64) < 0.01,
            "raiz medida em {} MB pelas duas chamadas, {} MB pela outra",
            root.total_mb,
            reference.total_mb
        );
    }

    /// O acelerador da Apple ou responde as duas medidas, ou não responde
    /// nenhuma. O que não pode acontecer é responder um número fora de escala.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn this_machine_accelerator_reports_plausible_usage() {
        let Some(stats) = mac_accelerator_stats().await else {
            return;
        };
        if let Some(uso) = stats.utilization_pct {
            assert!((0.0..=100.0).contains(&uso), "uso da GPU em {uso}%");
        }
        if let Some(mb) = stats.in_use_memory_mb {
            assert!(mb >= 0.0);
        }
    }

    /// O caso do captain: o executável rodando de uma pasta de rede não pode
    /// fazer todo peer relatar o disco da máquina que hospeda o
    /// compartilhamento.
    #[cfg(windows)]
    #[test]
    fn a_unc_path_is_recognized_as_network() {
        assert!(win::is_network_path(Path::new(
            r"\\servidor\compartilhamento\pom.exe"
        )));
    }

    #[cfg(windows)]
    #[test]
    fn the_local_system_drive_is_never_empty() {
        let drive = win::local_system_drive();
        assert!(!drive.as_os_str().is_empty());
    }

    #[tokio::test]
    async fn external_command_output_uses_the_blocking_process_path() {
        let output = command_output("printf", vec!["ok".into()])
            .await
            .expect("printf should run");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"ok");
    }

    #[test]
    fn current_process_gpu_memory_sums_only_the_requested_pid() {
        let output = "100, 12\n42, 700\n42, 30\n";

        assert_eq!(parse_current_process_gpu_memory_mb(output, 42), Some(730.0));
        assert_eq!(parse_current_process_gpu_memory_mb(output, 101), None);
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn cpu_only_backend_inventory_serializes_metadata_without_a_null_string() {
        // A enumeração é deliberadamente feita sem inicializar o backend: o
        // sysmon chama esta função depois da inicialização normal, mas o
        // registro de devices já é observável aqui e o teste não deve ligar
        // Metal/CUDA de uma máquina de desenvolvimento.
        let mut raw = ptr::null_mut();
        let status = unsafe { llama_cpp_sys_2::llama_rs_backend_devices_json(&mut raw) };
        assert_eq!(status, llama_cpp_sys_2::LLAMA_RS_STATUS_OK);
        assert!(
            !raw.is_null(),
            "a serialização bem-sucedida deve devolver JSON"
        );

        let text = unsafe { CStr::from_ptr(raw).to_string_lossy().into_owned() };
        unsafe { llama_cpp_sys_2::llama_rs_string_free(raw) };

        let entries: serde_json::Value =
            serde_json::from_str(&text).expect("a lista de devices deve ser JSON válido");
        let entries = entries
            .as_array()
            .expect("a lista de devices deve ser um array");
        for entry in entries {
            assert!(
                entry
                    .get("backend")
                    .and_then(serde_json::Value::as_str)
                    .is_some(),
                "backend nulo ou não textual: {entry}"
            );
            assert!(
                entry
                    .get("description")
                    .and_then(serde_json::Value::as_str)
                    .is_some(),
                "description nula ou não textual: {entry}"
            );
        }
    }

    #[test]
    fn an_inventory_without_gpu_preserves_the_empty_list() {
        assert!(parse_backend_gpu_devices("[]").is_empty());
    }
}

/// Traduz a escolha de GPUs da `pom.json` para o que cada backend do ggml lê.
///
/// **Precisa rodar antes do primeiro `llama_backend_init`.** Os backends
/// enumeram os dispositivos ao serem registrados, e a partir daí a lista está
/// congelada: mexer nas variáveis depois não tira nem devolve placa nenhuma.
///
/// A tradução existe porque os índices vivem em espaços diferentes. O nó publica
/// uma lista só (`system.gpus[]`, NVIDIA primeiro, Radeon depois) e é esse
/// número que o operador vê no painel e escreve na config; já o CUDA numera só
/// as placas NVIDIA e o Vulkan só as dele. Mandar o índice do nó direto para o
/// backend selecionaria a placa errada assim que houvesse duas marcas.
///
/// Devolve o que aplicou, para o log dizer em que placas o nó vai rodar.
/// A escolha de GPUs em vigor, para `read_gpu` filtrar o que reporta.
///
/// `None` = sem restrição, que cobre tanto "a config não pediu nada" quanto
/// "a config pediu todas" - nos dois casos não há o que filtrar.
static SELECTION: std::sync::RwLock<Option<Vec<u32>>> = std::sync::RwLock::new(None);

pub fn apply_gpu_selection(selection: Option<&[u32]>, detected: &[GpuStat]) -> Vec<String> {
    // Guardado antes de qualquer saída: o que o nó **reporta** tem que ser o que
    // ele pode **usar**. Não é cosmética - `deploy::memory_capacity` soma a VRAM
    // de `system.gpus[]` para decidir quantas camadas cabem, então um nó preso a
    // uma placa que anunciasse as duas receberia trabalho que não cabe nele.
    if let Ok(mut guard) = SELECTION.write() {
        *guard = selection.map(|s| s.to_vec());
    }

    let Some(selection) = selection else {
        // Ausente é "todas": não tocar nas variáveis é justamente o que deixa
        // cada backend enxergar tudo o que existe.
        return Vec::new();
    };

    // Índice do nó -> índice dentro do próprio fabricante.
    let mut cuda = Vec::new();
    let mut vulkan = Vec::new();
    let (mut n_nvidia, mut n_amd) = (0u32, 0u32);
    let mut chosen = Vec::new();

    for gpu in detected {
        let local = match gpu.vendor {
            GpuVendor::Nvidia => {
                let i = n_nvidia;
                n_nvidia += 1;
                i
            }
            GpuVendor::Amd => {
                let i = n_amd;
                n_amd += 1;
                i
            }
            // Apple não tem seleção por dispositivo: o Metal serve a única GPU
            // da máquina, e não há variável para restringir.
            GpuVendor::Apple => continue,
        };
        if !selection.contains(&gpu.index) {
            continue;
        }
        chosen.push(format!("{} (#{})", gpu.name, gpu.index));
        match gpu.vendor {
            GpuVendor::Nvidia => cuda.push(local.to_string()),
            GpuVendor::Amd => vulkan.push(local.to_string()),
            GpuVendor::Apple => {}
        }
    }

    // Lista vazia é escolha explícita por CPU, então as variáveis vão vazias -
    // que é como CUDA e Vulkan dizem "nenhum dispositivo".
    std::env::set_var("CUDA_VISIBLE_DEVICES", cuda.join(","));
    std::env::set_var("GGML_VK_VISIBLE_DEVICES", vulkan.join(","));

    // Sem isto o índice de cima está errado, e erra em silêncio.
    //
    // `GGML_VK_VISIBLE_DEVICES` indexa a lista de dispositivos **do Vulkan**, e
    // o Vulkan enumera a NVIDIA junto com a Radeon: o driver da NVIDIA também
    // publica um ICD. Contamos o índice entre as AMD, então pedir a Radeon podia
    // selecionar a NVIDIA - foi o que aconteceu na ksia, com o modelo indo para
    // a placa errada enquanto o log dizia `Vulkan0`.
    //
    // Restringir o *loader* a um ICD é o que torna a conta certa por
    // construção: com só a Radeon exposta, o índice entre as AMD **é** o índice
    // do Vulkan. A NVIDIA não perde nada - quem a serve é o CUDA.
    #[cfg(target_os = "linux")]
    if !vulkan.is_empty() {
        if let Some(icd) = icd_amd() {
            std::env::set_var("VK_DRIVER_FILES", &icd);
            // O nome antigo, para loaders anteriores ao 1.3.207.
            std::env::set_var("VK_ICD_FILENAMES", &icd);
        }
    }

    chosen
}

/// O ICD do Mesa/RADV, quando existe nesta máquina.
///
/// Caminho fixo porque é onde o Mesa instala em toda distribuição que empacota
/// Vulkan; achar nada apenas deixa o loader com o comportamento padrão, que é o
/// que já havia antes. O nome varia com a arquitetura (`radeon_icd.x86_64.json`),
/// daí a varredura em vez de um nome literal.
#[cfg(target_os = "linux")]
fn icd_amd() -> Option<String> {
    let dir = std::fs::read_dir("/usr/share/vulkan/icd.d").ok()?;
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("radeon_icd") && name.ends_with(".json") {
            return Some(entry.path().to_string_lossy().to_string());
        }
    }
    None
}

#[cfg(test)]
mod gpu_selection_tests {
    use super::*;

    /// As variáveis de ambiente são do processo inteiro, e o cargo roda os
    /// testes em paralelo: sem este mutex um teste lê o que o outro acabou de
    /// escrever. Serializar é o que torna o resultado determinístico.
    static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn gpu(index: u32, vendor: GpuVendor) -> GpuStat {
        GpuStat {
            index,
            name: format!("gpu{index}"),
            vendor,
            vram_total_mb: None,
            vram_used_mb: None,
            vram_used_pct: None,
            utilization_pct: None,
            power_watts: None,
            temperature_c: None,
        }
    }

    fn both() -> Vec<GpuStat> {
        vec![
            gpu(0, GpuVendor::Nvidia),
            gpu(1, GpuVendor::Amd),
            gpu(2, GpuVendor::Amd),
        ]
    }

    /// Sem escolha na config, nada é restringido - é o padrão pedido: todas.
    #[test]
    fn absent_restricts_nothing() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        std::env::remove_var("GGML_VK_VISIBLE_DEVICES");
        assert!(apply_gpu_selection(None, &both()).is_empty());
        assert!(std::env::var("CUDA_VISIBLE_DEVICES").is_err());
        assert!(std::env::var("GGML_VK_VISIBLE_DEVICES").is_err());
    }

    /// O ponto todo da tradução: a segunda Radeon é `#2` para o nó e `1` para o
    /// Vulkan. Mandar `2` selecionaria uma placa que não existe naquele backend.
    #[test]
    fn the_node_index_becomes_the_backend_index() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        apply_gpu_selection(Some(&[2]), &both());
        assert_eq!(std::env::var("GGML_VK_VISIBLE_DEVICES").unwrap(), "1");
        assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").unwrap(), "");
    }

    #[test]
    fn choosing_the_nvidia_does_not_release_the_radeons() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        apply_gpu_selection(Some(&[0]), &both());
        assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").unwrap(), "0");
        assert_eq!(std::env::var("GGML_VK_VISIBLE_DEVICES").unwrap(), "");
    }

    /// O nó tem que **reportar** o que pode **usar**.
    ///
    /// Não é cosmética de painel: `deploy::memory_capacity` soma o
    /// `vram_total_mb` de `system.gpus[]` para decidir quantas camadas cabem num
    /// nó. Um nó presa à Radeon de 4 GB que anunciasse as duas placas diria ter
    /// 20 GB, e receberia trabalho que não cabe nele.
    #[test]
    fn the_node_reports_only_what_it_may_use() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        apply_gpu_selection(Some(&[1]), &both());

        let mut visible = both();
        if let Ok(choice) = SELECTION.read() {
            if let Some(allowed) = choice.as_ref() {
                visible.retain(|g| allowed.contains(&g.index));
            }
        }
        assert_eq!(visible.len(), 1, "so a placa escolhida deve ser reportada");
        // O indice original sobrevive: e por ele que a config e o painel falam
        // desta placa, e renumerar faria `gpus: [1]` aparecer como `#0`.
        assert_eq!(visible[0].index, 1);
    }

    /// Sem restricao nada e filtrado - o padrao continua mostrando tudo.
    #[test]
    fn without_a_restriction_reports_all() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        apply_gpu_selection(None, &both());
        let visible = SELECTION.read().unwrap().clone();
        assert!(visible.is_none(), "sem escolha nao deve haver filtro");
    }

    /// Lista vazia é "só CPU", e não "não configurado".
    #[test]
    fn an_empty_list_turns_off_both_families() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        apply_gpu_selection(Some(&[]), &both());
        assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").unwrap(), "");
        assert_eq!(std::env::var("GGML_VK_VISIBLE_DEVICES").unwrap(), "");
    }

    #[test]
    fn the_backend_description_takes_precedence_over_pci_ids() {
        assert_eq!(
            resolve_gpu_name(
                Some("AMD Radeon RX 7800 XT (RADV NAVI32)"),
                Some("Navi 32 [Radeon RX 7700 XT / 7800 XT]"),
                "AMD Radeon (0x747e)".to_string(),
            ),
            "AMD Radeon RX 7800 XT (RADV NAVI32)"
        );
    }

    #[test]
    fn pci_ids_is_used_when_the_backend_does_not_answer() {
        assert_eq!(
            resolve_gpu_name(
                None,
                Some("Navi 32 [Radeon RX 7700 XT / 7800 XT]"),
                "AMD Radeon (0x747e)".to_string(),
            ),
            "Navi 32 [Radeon RX 7700 XT / 7800 XT]"
        );
    }

    #[test]
    fn the_identifier_fallback_never_removes_the_card() {
        assert_eq!(
            resolve_gpu_name(None, None, "AMD Radeon (0x747e)".to_string()),
            "AMD Radeon (0x747e)"
        );
    }

    #[test]
    fn pci_ids_resolves_the_right_name_for_each_card() {
        let entries = vec![
            BackendGpu {
                backend: "Vulkan".to_string(),
                description: "Radeon A".to_string(),
                device_id: Some("0000:03:00.0".to_string()),
            },
            BackendGpu {
                backend: "Vulkan".to_string(),
                description: "Radeon B".to_string(),
                device_id: Some("0000:0b:00.0".to_string()),
            },
        ];

        assert_eq!(
            backend_gpu_name(&entries, "Vulkan", Some("0000:03:00.0")),
            Some("Radeon A")
        );
        assert_eq!(
            backend_gpu_name(&entries, "Vulkan", Some("0000:0b:00.0")),
            Some("Radeon B")
        );
    }

    #[test]
    fn pci_ids_parser_reads_the_device_name_without_the_product_table() {
        let pci_ids =
            "1002  Advanced Micro Devices, Inc.\n\t747e  Navi 32 [Radeon RX 7700 XT / 7800 XT]\n";
        assert_eq!(
            pci_device_name(pci_ids, "1002", "747e"),
            Some("Navi 32 [Radeon RX 7700 XT / 7800 XT]".to_string())
        );
    }

    #[test]
    fn system_profiler_reads_the_gpu_name_and_not_the_cpu_name() {
        assert_eq!(
            system_profiler_gpu_name(
                "Graphics/Displays:\n\n    Apple M4:\n      Chipset Model: Apple M4\n"
            ),
            Some("Apple M4".to_string())
        );
    }

    /// O nome do dispositivo pode ter espaço (`map auto_home`), e é por isso
    /// que a linha é lida da direita. Lida da esquerda, o ponto de montagem
    /// desta linha viraria "0" e o total viraria "auto_home".
    #[test]
    fn df_reads_the_line_from_right_to_left() {
        let disk = parse_df_line("/dev/disk3s1s1  482797652  16686060  17602564  49%  /").unwrap();
        assert_eq!(disk.device, "/dev/disk3s1s1");
        // KiB do `df -k` viram MB aqui: 482797652 KiB = 471480 MB.
        assert_eq!(disk.total_mb, 471_482);
        assert_eq!(disk.free_mb, 17_190);
        assert_eq!(disk.mount_point, "/");

        let with_space =
            parse_df_line("map auto_home  0  0  0  100%  /System/Volumes/home").unwrap();
        assert_eq!(with_space.device, "map auto_home");
        assert_eq!(with_space.mount_point, "/System/Volumes/home");
    }

    /// Um `tmpfs` é RAM e um `devfs` não guarda byte: listá-los prometeria
    /// espaço que não existe. A mesma montagem duas vezes também não entra.
    #[test]
    fn df_lists_only_real_volumes_and_without_repeats() {
        let output = "Filesystem 1048576-blocks Used Available Capacity Mounted on\n\
             /dev/nvme0n1p2 500000 100000 400000 20% /\n\
             devfs 0 0 0 100% /dev\n\
             tmpfs 16000 0 16000 0% /dev/shm\n\
             /dev/disk3s1s2 500000 100000 400000 20% /System/Volumes/Preboot\n\
             /dev/disk3s1s3 500000 100000 400000 20% /System/Volumes/VM\n\
             /dev/disk3s1s4 500000 100000 400000 20% /System/Volumes/Update/mnt1\n\
             /dev/nvme0n1p1 500000 100000 400000 20% /boot/efi\n\
             /dev/sda1 2000000 1000000 1000000 50% /dados\n\
             /dev/sda1 2000000 1000000 1000000 50% /dados\n";
        let disks = parse_df_all(output);
        assert_eq!(
            disks
                .iter()
                .map(|d| d.mount_point.as_str())
                .collect::<Vec<_>>(),
            vec!["/", "/dados"]
        );
        assert_eq!(disks[1].device, "/dev/sda1");
    }

    #[test]
    fn the_macos_data_mount_stays_visible() {
        assert!(is_user_data_mount("/System/Volumes/Data"));
        assert!(is_user_data_mount("/Volumes/External"));
        assert!(!is_user_data_mount("/System/Volumes/iSCPreboot"));
        assert!(!is_user_data_mount("/Volumes/Recovery"));
    }

    /// A GPU da Apple não tem VRAM para medir, mas publica uso e memória em
    /// mão. Campo ausente continua `None`: "0%" e "não perguntei" são
    /// respostas diferentes.
    #[test]
    fn ioreg_reads_the_accelerator_usage_and_memory() {
        let output = "+-o AGXAcceleratorG16_B0  <class AGXAcceleratorG16_B0>\n    {\n\
             \"PerformanceStatistics\" = {\"Alloc system memory\"=1610612736,\"Device Utilization %\"=37,\"In use system memory\"=1073741824}\n    }\n";
        let stats = parse_ioreg_accelerator(output);
        assert_eq!(stats.utilization_pct, Some(37.0));
        assert_eq!(stats.in_use_memory_mb, Some(1024.0));

        assert_eq!(
            parse_ioreg_accelerator("sem nada"),
            MacAcceleratorStats::default()
        );
    }

    /// O `fdinfo` do amdgpu escreve a unidade junto do número. Lê-lo sem a
    /// unidade transformaria 1 MiB em 1 byte.
    #[test]
    fn amdgpu_fdinfo_counts_the_vram_with_its_unit() {
        let fdinfo = "pos:\t0\nflags:\t02100002\n\
             drm-driver:\tamdgpu\n\
             drm-client-id:\t12\n\
             drm-memory-vram:\t2048 KiB\n\
             drm-memory-gtt:\t256 KiB\n";
        assert_eq!(
            parse_amdgpu_fdinfo(fdinfo),
            Some(("12".to_string(), 2 * 1024 * 1024))
        );

        // `drm-resident-vram` é o nome novo e vence onde os dois existem.
        let new = "drm-driver:\tamdgpu\ndrm-client-id:\t7\n\
             drm-resident-vram:\t1 MiB\ndrm-memory-vram:\t9 MiB\n";
        assert_eq!(
            parse_amdgpu_fdinfo(new),
            Some(("7".to_string(), 1024 * 1024))
        );

        // Descritor que não é do amdgpu não entra na conta.
        assert_eq!(
            parse_amdgpu_fdinfo("drm-driver:\ti915\ndrm-client-id:\t1\n"),
            None
        );
        assert_eq!(parse_amdgpu_fdinfo("pos:\t0\n"), None);
    }
}
