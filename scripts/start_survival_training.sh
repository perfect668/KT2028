#!/usr/bin/env bash
# Linux launcher: detached training, durable logs, foreground viewer only.
set -euo pipefail
umask 077

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname -- "$REPO")"
RUN_NAME=""
COHORT_NAME="surgen413"
MODE="start"
FOLLOW=1
die() { printf '%s\n' "$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --check) MODE="check"; shift ;;
    --attach) MODE="attach"; shift ;;
    --no-follow) FOLLOW=0; shift ;;
    --project-root) (($# >= 2)) || die '缺少project-root'; ROOT="$2"; shift 2 ;;
    --run-name) (($# >= 2)) || die '缺少run-name'; RUN_NAME="$2"; shift 2 ;;
    --cohort) (($# >= 2)) || die '缺少cohort'; COHORT_NAME="$2"; shift 2 ;;
    *) die "未知参数：$1" ;;
  esac
done
case "$COHORT_NAME" in
  surgen413) DEFAULT_RUN="survival_s41_413_20261014"; MODULE="kt2028.train_survival" ;;
  tcga) DEFAULT_RUN="tcga_survival_s41_20261021"; MODULE="kt2028.train_mixed_survival" ;;
  *) die 'cohort仅支持surgen413或tcga' ;;
esac
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN}"
[[ "$ROOT" == /* ]] || die 'project-root须为服务器绝对路径'
[[ "$RUN_NAME" =~ ^[[:alnum:]_][[:alnum:]_-]*$ ]] || die 'run-name仅允许字母数字下划线和连字符'
ROOT="${ROOT%/}"
REPO="$ROOT/repo"
PYTHON="$ROOT/.venv/bin/python"
if [[ "$COHORT_NAME" == tcga ]]; then
  COHORT="$ROOT/data/tcga_survival_w069_20261021"
  CONFIG="$REPO/configs/predictor_tcga_survival.json"
  TRAIN_MANIFEST="$COHORT/train_manifest.json"
  VALIDATION_MANIFEST="$COHORT/validate_manifest.json"
else
  COHORT="$ROOT/audit/survival_preparation_w046_20261014/cohort413"
  CONFIG="$REPO/configs/predictor_survival.json"
  TRAIN_MANIFEST="$COHORT/train_survival_manifest.json"
  VALIDATION_MANIFEST="$COHORT/validate_survival_manifest.json"
fi
RUN="$ROOT/runs/$RUN_NAME"
CONTROL="$ROOT/launches/$RUN_NAME"
LOG="$CONTROL/training.log"
PIDFILE="$CONTROL/pid"

watch_log() {
  printf '日志：%s\n按Ctrl+C只退出日志查看，后台训练继续。\n' "$LOG"
  exec tail -n 80 -F -- "$LOG"
}
if [[ "$MODE" == attach ]]; then
  [[ -f "$LOG" ]] || die "尚无日志：$LOG"
  watch_log
fi
for tool in flock setsid nohup tail tr grep git; do
  command -v "$tool" >/dev/null || die "服务器缺少工具：$tool"
done
[[ -x "$PYTHON" && -d "$REPO/.git" ]] || die 'Python或服务器repo路径不可用'
[[ -f "$TRAIN_MANIFEST" && -f "$VALIDATION_MANIFEST" ]] || die '患者清单缺失'
[[ -f "$CONFIG" ]] || die '生存配置缺失'
cd -- "$REPO"

active_pid() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat -- "$PIDFILE")"
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fx "$MODULE" >/dev/null &&
    tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fx -- "$RUN" >/dev/null
}
if active_pid; then
  printf '已有此训练进程PID=%s，不重复启动。\n' "$(cat -- "$PIDFILE")"
  [[ "$MODE" != check && "$FOLLOW" == 1 ]] && watch_log
  exit 0
fi
[[ ! -e "$RUN" ]] || die "已有run，保留且不覆盖：$RUN"
[[ ! -e "$LOG" && ! -e "$PIDFILE" ]] || die '已有启动记录，先用--attach查看；新任务须明确指定不同--run-name'
[[ -z "$(git status --porcelain)" ]] || die 'repo存在未提交修改；先保存修改再启动'
CMD=("$PYTHON" -u -m "$MODULE"
  --train-manifest "$TRAIN_MANIFEST"
  --validation-manifest "$VALIDATION_MANIFEST"
  --config "$CONFIG"
  --output "$RUN" --device cuda --execute)
if [[ "$COHORT_NAME" == tcga ]]; then
  CMD+=(--tcga-features "$ROOT/data/tcga_w059/features"
        --preflight-report "$CONTROL/preflight.json")
else
  CMD+=(--regions "$ROOT/data/coarse_uni_r2_20261013")
fi
if [[ "$MODE" == check ]]; then
  printf '启动路径/工具/源状态检查通过；此检查不启动进程。\n输出：%s\n日志：%s\n' "$RUN" "$LOG"
  printf '拟执行命令：'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p -- "$CONTROL"
exec 9>"$CONTROL/launch.lock"
flock -n 9 || die '启动锁已占用；使用--attach查看当前日志'
# Check again under the lock; protects concurrent terminal invocations.
[[ ! -e "$RUN" && ! -e "$LOG" && ! -e "$PIDFILE" ]] || die '已有启动/训练记录；不覆盖或重复启动'
export TZ=Asia/Shanghai
export PYTHONUNBUFFERED=1
set -o noclobber
# nohup ignores SIGHUP; setsid severs the SSH session; stdin/out/err are detached.
# FD9 is inherited by the child, so the lock lives for the entire training process.
nohup setsid "${CMD[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$PIDFILE"
printf '已提交后台启动，PID=%s；先核对输入，随后开始训练。\n' "$PID"
printf '训练输出：%s\n重新连接后：bash %q --cohort %q --project-root %q --run-name %q --attach\n' "$RUN" "$REPO/scripts/start_survival_training.sh" "$COHORT_NAME" "$ROOT" "$RUN_NAME"
exec 9>&-
if [[ "$FOLLOW" == 1 ]]; then
  watch_log
fi
