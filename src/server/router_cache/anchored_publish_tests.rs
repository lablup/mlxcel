use super::*;
use std::ffi::OsStr;
use std::io::Write;
use std::os::fd::FromRawFd;
use std::os::unix::ffi::OsStrExt;

#[test]
fn publish_refuses_destination_created_after_stage() {
    let root = tempfile::tempdir().expect("root");
    let mut stage = AnchoredStage::create(root.path(), "owner/model").expect("stage");
    fs::create_dir_all(root.path().join("owner/model")).expect("raced dest");
    let err = stage.publish().expect_err("destination race must fail");
    assert!(err.to_string().contains("destination appeared"));
}

#[test]
fn publish_refuses_owner_directory_identity_swap() {
    let root = tempfile::tempdir().expect("root");
    let mut stage = AnchoredStage::create(root.path(), "owner/model").expect("stage");
    fs::rename(root.path().join("owner"), root.path().join("owner-old")).expect("move owner");
    fs::create_dir(root.path().join("owner")).expect("replacement owner");
    let err = stage.publish().expect_err("identity swap must fail");
    assert!(err.to_string().contains("owner directory identity changed"));
    assert!(!root.path().join("owner/model").exists());
}

#[test]
fn cleanup_refuses_staging_parent_identity_swap() {
    let root = tempfile::tempdir().expect("root");
    let mut stage = AnchoredStage::create(root.path(), "owner/model").expect("stage");
    let moved = root.path().join("staging-old");
    fs::rename(root.path().join(STAGING_DIR), &moved).expect("move staging parent");
    fs::create_dir(root.path().join(STAGING_DIR)).expect("replacement staging parent");
    fs::create_dir(root.path().join(STAGING_DIR).join("attacker")).expect("attacker");
    let old_stage_name = OsStr::from_bytes(stage.stage_name.to_bytes()).to_owned();
    stage.cleanup();
    assert!(root.path().join(STAGING_DIR).join("attacker").exists());
    assert!(moved.join(old_stage_name).exists());
}

#[test]
fn cleanup_unlinks_symlink_without_deleting_target() {
    let root = tempfile::tempdir().expect("root");
    let mut stage = AnchoredStage::create(root.path(), "owner/model").expect("stage");
    let victim = root.path().join("victim");
    fs::write(&victim, b"keep").expect("victim");
    let fd = unsafe {
        libc::openat(
            stage.stage_fd(),
            c"config.json".as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
            0o600,
        )
    };
    assert!(fd >= 0, "openat config.json failed");
    let mut file = unsafe { File::from_raw_fd(fd) };
    writeln!(file, "{{}}").expect("write");
    let victim_c = std::ffi::CString::new(victim.as_os_str().as_bytes()).expect("victim path");
    let rc = unsafe { libc::symlinkat(victim_c.as_ptr(), stage.stage_fd(), c"link".as_ptr()) };
    assert_eq!(rc, 0, "symlinkat failed");
    stage.cleanup();
    assert_eq!(fs::read(&victim).expect("victim remains"), b"keep");
}
