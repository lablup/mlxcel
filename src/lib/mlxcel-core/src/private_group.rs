// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Whether the process's effective group is provably private to the current
//! user, so a directory above a cache root that only that group can write
//! (`~/.cache` under a umask of 002 with user private groups) cannot be used
//! by anyone else (#3248).
//!
//! The answer is read from plain-text `/etc/nsswitch.conf`, `/etc/group` and
//! `/etc/passwd`, never from `getpwent`, because a group is only provably
//! private when every account and membership comes from local files:
//!
//! - `nsswitch.conf` lists only `files`, `systemd` or `compat` for `passwd`
//!   and `group` (and `initgroups`, if present), on every line naming them,
//!   parsed the way glibc does. Any other source (`sss`, `ldap`, `nis`, `winbind`,
//!   ...) can add members that local files do not show. A missing or
//!   unreadable file is refused too, although glibc would then default to
//!   `files`: a host without the file is rare enough that failing closed
//!   costs nothing worth the special case.
//! - no line in either file is a `+` / `-` compat inclusion (which pulls
//!   entries from NIS);
//! - every `/etc/group` line with that gid has an empty member list or one
//!   holding only the current user's name;
//! - every `/etc/passwd` account whose primary gid is that gid is the current
//!   user (same uid).
//!
//! Anything unparsable is a refusal. The parsers take file contents and the
//! uid, gid and name as parameters so tests can drive them without root.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

/// Upper bound on each file read; a larger file is refused rather than
/// parsed partially.
const MAX_FILE_BYTES: u64 = 4 << 20;

/// Name-service sources that resolve only from the local host.
const LOCAL_SOURCES: [&str; 3] = ["files", "systemd", "compat"];

/// Whether the source list of one `nsswitch.conf` line names only local
/// sources. Bracketed actions such as `[NOTFOUND=return]` (which may hold
/// spaces) are not sources and are skipped; an unterminated bracket, an
/// unknown source, or an empty list is not local.
fn sources_are_local(sources: &str) -> bool {
    let mut any = false;
    let mut in_action = false;
    for token in sources.split_whitespace() {
        if in_action {
            in_action = !token.ends_with(']');
            continue;
        }
        if token.starts_with('[') {
            in_action = !token.ends_with(']');
            continue;
        }
        if !LOCAL_SOURCES.contains(&token) {
            return false;
        }
        any = true;
    }
    any && !in_action
}

/// Whether `contents` (an `nsswitch.conf`) resolves `passwd`, `group` and
/// `initgroups` from local sources only.
///
/// This follows glibc's `process_line` (`nss/nss_database.c`): leading white
/// space is skipped, the database name is the leading token ended by a colon
/// or white space (the colon is not required), and a later line for the same
/// database overrides an earlier one. Because an override would hide a remote
/// source on an earlier line, every line naming a checked database must list
/// only local sources, not just the last. A line whose name or source list
/// cannot be parsed (a checked name with no sources) refuses, where glibc
/// would skip it. `passwd` and `group` must both be present; `initgroups` is
/// optional. When `compat` is listed, glibc reads the sources from
/// `passwd_compat` / `group_compat`, so those lines are held to the same rule.
pub(crate) fn nsswitch_is_local(contents: &str) -> bool {
    let mut seen = [false; 2];
    for line in contents.lines() {
        let line = line.split('#').next().unwrap_or("").trim_start();
        let name_end = line
            .find(|c: char| c.is_whitespace() || c == ':')
            .unwrap_or(line.len());
        let name = &line[..name_end];
        let checked = matches!(
            name,
            "passwd" | "group" | "initgroups" | "passwd_compat" | "group_compat"
        );
        if !checked {
            continue;
        }
        let sources = line[name_end..].trim_start_matches(|c: char| c.is_whitespace() || c == ':');
        if !sources_are_local(sources) {
            return false;
        }
        match name {
            "passwd" => seen[0] = true,
            "group" => seen[1] = true,
            _ => {}
        }
    }
    seen == [true, true]
}

/// Whether a `/etc/passwd` or `/etc/group` line is a `+` / `-` compat
/// inclusion. glibc skips leading white space before the prefix, so so do we.
fn is_compat_line(line: &str) -> bool {
    line.trim_start().starts_with(['+', '-'])
}

/// Whether every `/etc/group` line for `gid` lists no member other than
/// `user`. A malformed line or a compat inclusion refuses.
pub(crate) fn group_members_only(contents: &str, gid: u32, user: &str) -> bool {
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        if is_compat_line(line) {
            return false;
        }
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() != 4 {
            return false;
        }
        let Ok(line_gid) = fields[2].parse::<u32>() else {
            return false;
        };
        if line_gid == gid
            && fields[3]
                .split(',')
                .map(str::trim)
                .any(|member| !member.is_empty() && member != user)
        {
            return false;
        }
    }
    true
}

/// Whether no `/etc/passwd` account other than `uid` has `gid` as its primary
/// group. A malformed line or a compat inclusion refuses.
pub(crate) fn no_other_primary_member(contents: &str, gid: u32, uid: u32) -> bool {
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        if is_compat_line(line) {
            return false;
        }
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() != 7 {
            return false;
        }
        let (Ok(line_uid), Ok(line_gid)) = (fields[2].parse::<u32>(), fields[3].parse::<u32>())
        else {
            return false;
        };
        if line_gid == gid && line_uid != uid {
            return false;
        }
    }
    true
}

/// The login name of `uid` in `/etc/passwd` contents (the first match).
pub(crate) fn user_name(contents: &str, uid: u32) -> Option<&str> {
    contents.lines().find_map(|line| {
        let fields: Vec<&str> = line.split(':').collect();
        (fields.len() == 7 && fields[2].parse::<u32>().ok() == Some(uid)).then_some(fields[0])
    })
}

/// The whole decision from file contents (`None` for a missing or unreadable
/// file): `gid` is private to `uid` (named `name`).
pub(crate) fn group_is_private(
    nsswitch: Option<&str>,
    group: Option<&str>,
    passwd: Option<&str>,
    uid: u32,
    gid: u32,
    name: &str,
) -> bool {
    let (Some(nsswitch), Some(group), Some(passwd)) = (nsswitch, group, passwd) else {
        return false;
    };
    !name.is_empty()
        && nsswitch_is_local(nsswitch)
        && group_members_only(group, gid, name)
        && no_other_primary_member(passwd, gid, uid)
}

/// Read `path` as text, refusing a file larger than [`MAX_FILE_BYTES`].
#[cfg(target_os = "linux")]
fn read_bounded(path: &str) -> Option<String> {
    use std::io::Read;
    let file = std::fs::File::open(path).ok()?;
    let mut contents = String::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_string(&mut contents)
        .ok()?;
    (contents.len() as u64 <= MAX_FILE_BYTES).then_some(contents)
}

/// Whether this process's effective gid is provably private to its effective
/// uid. Computed once per process; always `false` off Linux.
pub(crate) fn own_group_is_private() -> bool {
    #[cfg(target_os = "linux")]
    {
        static PRIVATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *PRIVATE.get_or_init(|| {
            // SAFETY: geteuid and getegid have no preconditions and cannot fail.
            let (uid, gid) = unsafe { (libc::geteuid(), libc::getegid()) };
            let passwd = read_bounded("/etc/passwd");
            let name = passwd
                .as_deref()
                .and_then(|contents| user_name(contents, uid))
                .unwrap_or("")
                .to_string();
            group_is_private(
                read_bounded("/etc/nsswitch.conf").as_deref(),
                read_bounded("/etc/group").as_deref(),
                passwd.as_deref(),
                uid,
                gid,
                &name,
            )
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = MAX_FILE_BYTES;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NSS_LOCAL: &str = "# comment\npasswd:         files systemd\ngroup:          files [SUCCESS=merge] systemd\nshadow: files\nhosts: files dns\n";
    const PASSWD: &str =
        "root:x:0:0:root:/root:/bin/bash\ndevops:x:1001:1001::/home/devops:/bin/bash\n";
    const GROUP: &str = "root:x:0:\nsudo:x:27:devops\ndevops:x:1001:\n";

    fn decide(nss: Option<&str>, group: &str, passwd: &str) -> bool {
        group_is_private(nss, Some(group), Some(passwd), 1001, 1001, "devops")
    }

    #[test]
    fn a_sole_member_user_private_group_is_accepted() {
        assert!(decide(Some(NSS_LOCAL), GROUP, PASSWD));
        assert!(decide(Some(NSS_LOCAL), "devops:x:1001:devops\n", PASSWD));
        assert!(decide(
            Some("passwd: compat\ngroup: compat\n"),
            GROUP,
            PASSWD
        ));
        assert_eq!(user_name(PASSWD, 1001), Some("devops"));
    }

    #[test]
    fn another_member_in_the_member_list_is_refused() {
        assert!(!decide(
            Some(NSS_LOCAL),
            "devops:x:1001:devops,alice\n",
            PASSWD
        ));
        assert!(!decide(
            Some(NSS_LOCAL),
            "devops:x:1001:\nalias:x:1001:bob\n",
            PASSWD
        ));
    }

    #[test]
    fn another_account_with_the_gid_as_primary_group_is_refused() {
        let passwd = format!("{PASSWD}alice:x:1002:1001::/home/alice:/bin/sh\n");
        assert!(!decide(Some(NSS_LOCAL), GROUP, &passwd));
    }

    #[test]
    fn a_remote_name_service_source_is_refused() {
        for nss in [
            "passwd: files sss\ngroup: files\n",
            "passwd: files\ngroup: files ldap\n",
            "passwd: files\ngroup: files\ngroup: nis\n",
            "passwd: files winbind\ngroup: files winbind\n",
            "passwd: files\n",
            "passwd:\ngroup: files\n",
        ] {
            assert!(!decide(Some(nss), GROUP, PASSWD), "{nss}");
        }
    }

    #[test]
    fn a_later_remote_line_overrides_an_earlier_local_one() {
        for nss in [
            "passwd: files\npasswd sss\ngroup: files\n",
            "passwd: files\ngroup: files\ngroup sss\n",
            "passwd: files\ngroup: files\ninitgroups: files sss\n",
            "passwd: files\ngroup: files\ninitgroups files\ninitgroups: nis\n",
            "passwd: compat\ngroup: compat\npasswd_compat: sss\n",
        ] {
            assert!(!decide(Some(nss), GROUP, PASSWD), "{nss}");
        }
    }

    #[test]
    fn the_colon_after_the_database_name_is_optional() {
        assert!(decide(Some("passwd files\ngroup files\n"), GROUP, PASSWD));
        assert!(decide(
            Some("  passwd \t files systemd\n\tgroup:files\ninitgroups: files\n"),
            GROUP,
            PASSWD
        ));
        assert!(!decide(Some("passwd sss\ngroup files\n"), GROUP, PASSWD));
        assert!(!decide(Some("passwd files\ngroup sss\n"), GROUP, PASSWD));
        assert!(!decide(Some("passwd\ngroup files\n"), GROUP, PASSWD));
        assert!(!decide(
            Some("passwd files\ngroup files\ngroup\n"),
            GROUP,
            PASSWD
        ));
    }

    #[test]
    fn bracketed_actions_are_skipped_but_must_terminate() {
        assert!(decide(
            Some("passwd: files [NOTFOUND=return UNAVAIL=continue] systemd\ngroup: files\n"),
            GROUP,
            PASSWD
        ));
        assert!(!decide(
            Some("passwd: files [NOTFOUND=return\ngroup: files\n"),
            GROUP,
            PASSWD
        ));
    }

    #[test]
    fn leading_whitespace_compat_lines_are_refused() {
        assert!(!decide(Some(NSS_LOCAL), " +devops:x:1001:\n", PASSWD));
        assert!(!decide(Some(NSS_LOCAL), GROUP, "\t+devops::::::\n"));
        assert!(!decide(Some(NSS_LOCAL), " -devops:x:1001:\n", PASSWD));
    }

    #[test]
    fn a_missing_or_unreadable_file_is_refused() {
        assert!(!decide(None, GROUP, PASSWD));
        assert!(!group_is_private(
            Some(NSS_LOCAL),
            None,
            Some(PASSWD),
            1001,
            1001,
            "devops"
        ));
        assert!(!group_is_private(
            Some(NSS_LOCAL),
            Some(GROUP),
            None,
            1001,
            1001,
            "devops"
        ));
        assert!(!group_is_private(
            Some(NSS_LOCAL),
            Some(GROUP),
            Some(PASSWD),
            1001,
            1001,
            ""
        ));
    }

    #[test]
    fn compat_inclusions_and_malformed_lines_are_refused() {
        assert!(!decide(Some(NSS_LOCAL), "+:::\n", PASSWD));
        assert!(!decide(Some(NSS_LOCAL), GROUP, "+::::::\n"));
        assert!(!decide(Some(NSS_LOCAL), "devops:x:notanumber:\n", PASSWD));
        assert!(!decide(Some(NSS_LOCAL), GROUP, "devops:x:1001\n"));
    }
}
