#!/usr/bin/env bash
# GPU monitoring during eval runs.
# Usage: ./scripts/gpu_monitor.sh [interval_seconds] [output_dir]

INTERVAL=${1:-1}
OUTDIR=${2:-/home/modfi/models/vla_simu/logs/profiling}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${OUTDIR}/gpu_monitor_${TIMESTAMP}.csv"

mkdir -p "$OUTDIR"

echo "timestamp,gpu_id,temp_c,power_w,power_cap_w,mem_used_mb,mem_total_mb,gpu_util_pct,mem_util_pct,pstate" > "$LOGFILE"
echo "Logging to: $LOGFILE (Ctrl+C to stop)"

cleanup() {
    echo ""
    echo "Stopped. Log: $LOGFILE ($(( $(wc -l < "$LOGFILE") - 1 )) entries)"
    exit 0
}
trap cleanup INT TERM

while true; do
    NOW=$(date +"%Y-%m-%d %H:%M:%S")

    # Query only NVIDIA GPUs explicitly
    RAW=$(nvidia-smi -i 0,1 --query-gpu=index,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu,utilization.memory,pstate \
        --format=csv,noheader,nounits 2>/dev/null)

    # Log to CSV
    while IFS= read -r row; do
        echo "${NOW},${row}" >> "$LOGFILE"
    done <<< "$RAW"

    # Live display
    clear
    echo "=== GPU Monitor @ ${NOW} ==="
    echo ""
    printf "%-5s %6s %8s %18s %6s %6s %6s\n" "GPU" "Temp" "Power" "VRAM" "GPU%" "MEM%" "State"
    printf "%-5s %6s %8s %18s %6s %6s %6s\n" "-----" "------" "--------" "------------------" "------" "------" "------"

    while IFS= read -r row; do
        # Trim all spaces, split on comma
        idx=$(echo "$row" | cut -d',' -f1 | tr -d ' ')
        temp=$(echo "$row" | cut -d',' -f2 | tr -d ' ')
        pwr=$(echo "$row" | cut -d',' -f3 | tr -d ' ')
        pwr_cap=$(echo "$row" | cut -d',' -f4 | tr -d ' ')
        mem_u=$(echo "$row" | cut -d',' -f5 | tr -d ' ')
        mem_t=$(echo "$row" | cut -d',' -f6 | tr -d ' ')
        gpu_u=$(echo "$row" | cut -d',' -f7 | tr -d ' ')
        mem_pct=$(echo "$row" | cut -d',' -f8 | tr -d ' ')
        ps=$(echo "$row" | cut -d',' -f9 | tr -d ' ')

        printf "%-5s %4s°C %6sW %6s/%6s MB %4s%% %4s%% %5s\n" \
            "$idx" "$temp" "$pwr" "$mem_u" "$mem_t" "$gpu_u" "$mem_pct" "$ps"
    done <<< "$RAW"

    echo ""
    echo "--- Processes ---"
    nvidia-smi -i 0,1 --query-compute-apps=gpu_bus_id,pid,used_gpu_memory,process_name \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r bus pid mem pname; do
        bus=$(echo "$bus" | tr -d ' ')
        pid=$(echo "$pid" | tr -d ' ')
        mem=$(echo "$mem" | tr -d ' ')
        pname=$(echo "$pname" | tr -d ' ' | xargs -I{} basename {})
        # Map bus ID to GPU index
        if [[ "$bus" == "00000000:01:00.0" ]]; then
            gpu="0"
        elif [[ "$bus" == "00000000:04:00.0" ]]; then
            gpu="1"
        else
            gpu="?"
        fi
        printf "  GPU %s  PID %-8s %10s  %s\n" "$gpu" "$pid" "$mem" "$pname"
    done

    echo ""
    echo "Logging every ${INTERVAL}s to $LOGFILE"
    sleep "$INTERVAL"
done
