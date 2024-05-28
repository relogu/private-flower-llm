#!/bin/bash
# Usage: ./system_monitor.sh <PID of the process>
# It also uses environmental variables $RUN_UUID and $CONCURRENCY that are set in the parent script
# Output: pidstat-$RUN_UUID-$CONCURRENCY.dat with lines such as `1539689171 305m 2.0`, i.e. unix time - memory with m/g suffix - CPU load in %
# To plot the output, see https://gist.github.com/jakubholynet/931a3441982c833f5f8fcdcf54d05c91
export PID=$1

# Function to get a recursive list of PIDs
get_pids() {
	echo $1
	local children=$(pgrep -P $1)
	for inner_pid in $children; do
		get_pids $inner_pid
	done
}

echo "Timestamp,Memory,CPU,kB_rd/s,kB_wr/s,kB_ccwr/s,I/O Delay,USR_MS,SYTEM_MS,GUEST_MS" >pidstat-$RUN_UUID-$CONCURRENCY.csv

while true; do
	# Get recursively the list of PIDs of the children of the process
	list_of_pids=$(get_pids $PID)
	# Transform the list to separate by commas
	list_of_pids=$(echo $list_of_pids | tr ' ' ',')
	# echo "List of PIDs: $list_of_pids"
	# Concat the list of PIDs to the PID of the parent process
	list_of_pids="$PID,$list_of_pids"
	total_mem=0
	total_cpu=0
	total_kB_rd_s=0
	total_kB_wr_s=0
	total_kB_ccwr_s=0
	total_iodelay=0
	for inner_pid in $(echo $list_of_pids | tr ',' ' '); do
		output=$(pidstat -r -p $inner_pid 0 | tail -1)
		mem=$(echo $output | awk '{print $9}')
		cpuoutput=$(pidstat -u -p $inner_pid 0 | tail -1)
		cpu=$(echo $cpuoutput | awk '{print $9}')
		diskoutput=$(pidstat -d -p $inner_pid 0 | tail -1)
		kB_rd_s=$(echo $diskoutput | awk '{print $5}')
		kB_wr_s=$(echo $diskoutput | awk '{print $6}')
		kB_ccwr_s=$(echo $diskoutput | awk '{print $7}')
		iodelay=$(echo $diskoutput | awk '{print $8}')

		total_mem=$(echo "$total_mem + $mem" | bc)
		total_cpu=$(echo "$total_cpu + $cpu" | bc)
		total_kB_rd_s=$(echo "$total_kB_rd_s + $kB_rd_s" | bc)
		total_kB_wr_s=$(echo "$total_kB_wr_s + $kB_wr_s" | bc)
		total_kB_ccwr_s=$(echo "$total_kB_ccwr_s + $kB_ccwr_s" | bc)
		total_iodelay=$(echo "$total_iodelay + $iodelay" | bc)
	done
	timeoutput=$(pidstat -T ALL -u -p $PID 0 | tail -1)
	usr_ms=$(echo $timeoutput | awk '{print $5}')
	system_ms=$(echo $timeoutput | awk '{print $6}')
	guest_ms=$(echo $timeoutput | awk '{print $7}')

	echo "$(date +%s%3N),$total_mem,$total_cpu,$total_kB_rd_s,$total_kB_wr_s,$total_kB_ccwr_s,$total_iodelay,$usr_ms,$system_ms,$guest_ms" >>pidstat-$RUN_UUID-$CONCURRENCY.csv
done
