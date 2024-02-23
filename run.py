import argparse
import pandas
import subprocess
import os
import pathlib
import sqlalchemy

parser = argparse.ArgumentParser()
parser.add_argument('--op', type=str, default='sendrecv', choices=['sendrecv', 'gather', 'scatter', 'broadcast', 'reduce', 'all_gather', 'reduce_scatter', 'all_reduce', 'alltoall', 'hypercube'], help='Type of collective operation')
parser.add_argument('--inner_loop', type=int, default=20, help='Number of iterations in inner loop')
parser.add_argument('--device_ids', type=int, nargs='+', default=[0], help='GPU device ids to use')
parser.add_argument('--nccl_tests_dir', type=pathlib.Path, default='.', help='nccl_tests directory')
parser.add_argument('--disable_p2p', action='store_true', help='Disable P2P access between GPUs')
parser.add_argument('--output_dir', type=pathlib.Path, default='output', help='Output directory path')

args = parser.parse_args()

if not args.output_dir.exists():
    os.makedirs(args.output_dir)

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

op_map = {
    'sendrecv': 'sendrecv_perf',
    'gather': 'gather_perf',
    'scatter': 'scatter_perf',
    'broadcast': 'broadcast_perf',
    'reduce': 'reduce_perf',
    'reduce_scatter': 'reduce_scatter_perf',
    'all_gather': 'all_gather_perf',
    'all_reduce': 'all_reduce_perf',
    'alltoall': 'alltoall_perf',
    'hypercube': 'hypercube_perf',
}

def run(op):
    bin_path = f'{os.path.abspath(args.nccl_tests_dir)}/build/{op_map[op]}'
    assert os.path.exists(bin_path), f'Binary {bin_path} does not exist'
    cmd = [bin_path, '--nthreads', str(len(args.device_ids)), '--ngpus', '1', '--minbytes', '4', '--maxbytes', '1G', '--stepfactor', '2', '--blocking', '1', '--datatype', 'uint8', '--average', '2', '--check', '0', '--iters', str(args.inner_loop)]
    print(f'Running {args.op.upper()} Benchmark on {args.device_ids}...')
    print(' '.join(cmd))
    # run command
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.device_ids])
    if args.disable_p2p:
        env['NCCL_P2P_DISABLE'] = '1'
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    assert result.returncode == 0, f'Error running command: {result.stderr}'
    report = pandas.DataFrame(columns=['size', 'count', 'type', 'redop', 'root', 'ofp_time', 'ofp_algbw', 'ofp_busbw', 'ofp_wrong', 'ip_time', 'ip_algbw', 'ip_busbw', 'ip_wrong'])
    output_str = result.stdout
    lines = output_str.splitlines()
    for line in lines:
        if line.startswith(b'#'):
            line = line[1:].strip().decode('utf-8')
            if line.find('nThread') > 0 or line.find('Rank') > 0:
                print(line)
            continue
        values = line.split()
        if args.op == 'hypercube':
            if len(values) == 12:
                values.insert(3, 'na.')
        if len(values) == 13:
            report.loc[len(report)] = values

    report['size'] = report['size'].astype(int)
    report['count'] = report['count'].astype(int)
    report['type'] = report['type'].astype(str)
    report['redop'] = report['redop'].astype(str)
    report['root'] = report['root'].astype(int)
    report['ofp_time'] = report['ofp_time'].astype(float)
    report['ofp_algbw'] = report['ofp_algbw'].astype(float)
    report['ofp_busbw'] = report['ofp_busbw'].astype(float)
    report['ofp_wrong'] = report['ofp_wrong'].astype(str)
    report['ip_time'] = report['ip_time'].astype(float)
    report['ip_algbw'] = report['ip_algbw'].astype(float)
    report['ip_busbw'] = report['ip_busbw'].astype(float)
    report['ip_wrong'] = report['ip_wrong'].astype(str)

    ofp_alg_avg_throughput = report['ofp_algbw'].max() 
    ofp_bus_avg_throughput = report['ofp_busbw'].max() 
    ip_alg_avg_throughput = report['ip_algbw'].max()
    ip_bus_avg_throughput = report['ip_busbw'].max()

    ofp_avg_latency = report['ofp_time'].min()
    ip_avg_latency = report['ip_time'].min()

    pid = os.getpid()
    has_iommu = env.get('HAS_IOMMU', 'unknown') # on, off
    card_num = env.get('CARD_NUM', 'unknown')
    numa_info = subprocess.check_output(['numactl', '--show', str(pid)]).decode()
    report['timestamp'] = pandas.Timestamp.today().date()
    if env.get('NODE_IS_NUMA_NODE', 'False') == 'True':
        numa_info = subprocess.check_output(['numactl', '--show', str(pid)]).decode()
        report['node_id'] = numa_info.split('\n')[1].split(':')[1].strip()
    else:
        report['node_id'] = subprocess.check_output(['hostname', '-s']).decode().strip()
    report['iommu'] = has_iommu
    report['card_num'] = card_num
    report['op'] = args.op
    report['device_ids'] = str(args.device_ids)
    if args.disable_p2p:
        report['p2p'] = 'shm'
    else:
        report['p2p'] = 'p2p'
    report = report.reindex(columns=[ 'timestamp', 'node_id', 'iommu', 'card_num', 'op', 'device_ids', 'p2p', 'size', 'count', 'type', 'redop', 'root', 'ofp_time', 'ofp_algbw', 'ofp_busbw', 'ofp_wrong', 'ip_time', 'ip_algbw', 'ip_busbw', 'ip_wrong'])

    # print report
    print()
    print(report[['op', 'p2p', 'size', 'ofp_time', 'ofp_algbw', 'ofp_busbw', 'ip_time', 'ip_algbw', 'ip_busbw']].to_string(index=False))
    print(f'Out-of-Place Avg. Latency: {ofp_avg_latency} us')
    print(f'In-Place     Avg. Latency: {ip_avg_latency} us')
    print(f'Out-of-Place Avg. Alg Throughput: {ofp_alg_avg_throughput} GB/s')
    print(f'In-Place     Avg. Alg Throughput: {ip_alg_avg_throughput} GB/s')
    print(f'Out-of-Place Avg. Bus Throughput: {ofp_bus_avg_throughput} GB/s')
    print(f'In-Place     Avg. Bus Throughput: {ip_bus_avg_throughput} GB/s')

    # save csv report
    report.to_csv(f'{args.output_dir}/{args.op}.csv', index=False)
    
    # plot
    plot = report.plot(x='size', y=['ofp_algbw', 'ip_algbw'], logx=True, logy=False, title=f'{args.op.upper()} Benchmark', xlabel='Message Size (bytes)', ylabel='Throughput (GB/s)')
    plot.legend(['Out-of-Place', 'In-Place'])
    plot.set_xlabel('Message Size (bytes)')
    plot.set_ylabel('Throughput (GB/s)')
    plot.get_figure().savefig(f'{args.output_dir}/{args.op}_alg.png')   

    plot = report.plot(x='size', y=['ofp_busbw', 'ip_busbw'], logx=True, logy=False, title=f'{args.op.upper()} Benchmark', xlabel='Message Size (bytes)', ylabel='Throughput (GB/s)')
    plot.legend(['Out-of-Place', 'In-Place'])
    plot.set_xlabel('Message Size (bytes)')
    plot.set_ylabel('Throughput (GB/s)')
    plot.get_figure().savefig(f'{args.output_dir}/{args.op}_bus.png')   

def main():
    run(args.op)

if __name__ == '__main__':
    main()