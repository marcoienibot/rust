use camino::Utf8PathBuf;

use crate::environment::Environment;
use crate::exec::Bootstrap;
use crate::tests::run_tests;
use crate::timer::Timer;
use crate::training::{
    ClippyPGOProfile, LlvmPGOProfile, RustcPGOProfile, RustdocPGOProfile, gather_llvm_profiles,
};
use crate::{gather_rustc_pgo_profiles, prepare_training};

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum WindowsPhase {
    RustcPgo,
    LlvmPgo,
    Dist,
}

pub fn execute_pipeline(
    env: &Environment,
    timer: &mut Timer,
    dist_args: Vec<String>,
    phase: WindowsPhase,
) -> anyhow::Result<()> {
    match phase {
        WindowsPhase::RustcPgo => {
            prepare_training(env)?;
            timer.section("Rustc + rustdoc + cargo + clippy PGO", |stage| {
                gather_rustc_pgo_profiles(env, stage)
            })?;
        }
        WindowsPhase::LlvmPgo => {
            prepare_training(env)?;
            let profile_root = env.artifact_dir().join("llvm-pgo");

            // Static LLVM must be linked into a fresh rustc. Training does not need
            // rustc's own PGO profiles, so this can run alongside RustcPgo.
            timer.section("Build rustc with PGO instrumented LLVM", |stage| {
                Bootstrap::build(env).with_cargo().llvm_pgo_instrument(&profile_root).run(stage)
            })?;
            timer.section("Gather LLVM profiles", |_| gather_llvm_profiles(env, &profile_root))?;
        }
        WindowsPhase::Dist => {
            // Keep the profiles downloaded from the two training jobs. This runner
            // has no instrumented compiler or LLVM artifacts to accidentally reuse.
            let rustc_profile = RustcPGOProfile(profile_path(env, "rustc")?);
            let rustdoc_profile = RustdocPGOProfile(profile_path(env, "rustdoc")?);
            let llvm_profile = LlvmPGOProfile(profile_path(env, "llvm")?);
            let mut dist = Bootstrap::dist(env, &dist_args)
                .rustc_pgo_optimize(&rustc_profile)
                .cargo_pgo_optimize(&rustc_profile)
                .rustdoc_pgo_optimize(&rustdoc_profile)
                .llvm_pgo_optimize(Some(&llvm_profile));
            if !env.is_fast_try_build() {
                let clippy_profile = ClippyPGOProfile(profile_path(env, "clippy")?);
                dist = dist.clippy_pgo_optimize(&clippy_profile);
            }

            timer.section("Final build", |stage| dist.run(stage))?;
            if !env.is_fast_try_build() && env.run_tests() {
                timer.section("Run tests", |_| run_tests(env))?;
            }
        }
    }
    Ok(())
}

fn profile_path(env: &Environment, component: &str) -> anyhow::Result<Utf8PathBuf> {
    let path = env.artifact_dir().join(format!("{component}-pgo.profdata"));
    anyhow::ensure!(path.is_file(), "Missing PGO profile: {path}");
    Ok(path)
}
