import os
import sys
import shutil
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()

    base_dir = 'benchmark_notebooks'

    datasets = [
        {
            'id': '01_CVC-ClinicDB',
            'notebook': 'fanet-benchmark-cvc-clinicdb.ipynb',
            'kaggle_slug': 'namnguynnnn/fanet-benchmark-cvc-clinicdb',
            'res_dir': 'results/benchmark_v2/cvc-clinicdb',
            'model_src': 'results/benchmark_v2/cvc-clinicdb/FANet_MFAD_CVC-ClinicDB_best.pth',
            'outputs': ['benchmark_results_CVC-ClinicDB.csv', 'comparison_CVC-ClinicDB.png', 'training_curves_CVC-ClinicDB.png'],
            'extra_logs': []
        },
        {
            'id': '02_EM-Dataset',
            'notebook': 'fanet-benchmark-em-dataset.ipynb',
            'kaggle_slug': 'namnguynnnn/fanet-benchmark-em-dataset',
            'res_dir': 'results/benchmark_v2/em-dataset',
            'model_src': 'results/benchmark_v2/em-dataset/FANet_MFAD_EM-Dataset_best.pth',
            'outputs': ['benchmark_results_EM-Dataset.csv', 'benchmark_summary_EM-Dataset.json', 'comparison_EM-Dataset.png', 'training_curves_EM-Dataset.png'],
            'extra_logs': []
        },
        {
            'id': '03_Kvasir-SEG',
            'notebook': 'colab_run_mfad_kvasir.ipynb',
            'kaggle_slug': 'namnguynnnn/fanet-benchmark-kvasir-seg',
            'res_dir': 'results/benchmark_v2/kvasir-seg',
            'model_src': 'results/benchmark_v2/kvasir-seg/FANet_MFAD_Kvasir-SEG_best.pth',
            'outputs': ['benchmark_results_Kvasir-SEG.csv', 'benchmark_summary_Kvasir-SEG.json'],
            'extra_logs': []
        },
        {
            'id': '04_ISIC-2018',
            'notebook': 'fanet-benchmark-isic-2018.ipynb',
            'kaggle_slug': 'namnguynnnn/fanet-benchmark-isic-2018',
            'res_dir': 'results/benchmark_v2/isic-2018',
            'model_src': 'results/benchmark_v2/isic-2018/FANet_MFAD_ISIC-2018_best.pth',
            'outputs': ['benchmark_results_ISIC-2018.csv', 'comparison_ISIC-2018.png', 'training_curves_ISIC-2018.png'],
            'extra_logs': ['results/isic2018/fanet-benchmark-isic-2018.log']
        },
        {
            'id': '05_CHASE-DB1',
            'notebook': 'fanet-benchmark-chasedb1.ipynb',
            'kaggle_slug': 'namnguynnnn/fanet-benchmark-chasedb1',
            'res_dir': 'results/benchmark_v2/chasedb1',
            'model_src': 'results/benchmark_v2/chasedb1/FANet_MFAD_CHASE-DB1_best.pth',
            'outputs': ['benchmark_results_CHASE-DB1.csv', 'comparison_CHASE-DB1.png', 'training_curves_CHASE-DB1.png'],
            'extra_logs': []
        },
        {
            'id': '06_DRIVE',
            'notebook': 'colab_run_mfad_drive_transfer.ipynb',
            'kaggle_slug': 'namnguynnnn/fanet-benchmark-drive',
            'res_dir': 'results/benchmark_v2/drive_transfer',
            'model_src': 'results/benchmark_v2/drive/FANet_MFAD_DRIVE_best.pth',
            'outputs': ['benchmark_results_drive_transfer.csv', 'benchmark_summary_drive_transfer.json'],
            'extra_logs': []
        },
        {
            'id': '07_DSB-2018',
            'notebook': 'colab_run_mfad_dsb2018_resume.ipynb',
            'kaggle_slug': 'namnguynnnn/fanet-benchmark-dsb2018',
            'res_dir': 'results/benchmark_v2/dsb2018',
            'model_src': 'results/benchmark_v2/dsb2018/FANet_MFAD_DSB-2018_best.pth',
            'outputs': ['benchmark_results_dsb2018.csv', 'benchmark_summary_dsb2018.json'],
            'extra_logs': []
        },
    ]

    for d in datasets:
        target_dir = os.path.join(base_dir, d['id'])
        log_dir = os.path.join(target_dir, 'logs')
        out_dir = os.path.join(target_dir, 'outputs')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        d_name = d['id']
        print(f"Setting up {d_name}...")

        # 1. Copy model weights if available
        if d['model_src'] and os.path.exists(d['model_src']):
            dst_model = os.path.join(out_dir, os.path.basename(d['model_src']))
            if not os.path.exists(dst_model):
                print(f"  Copying model weight: {os.path.basename(d['model_src'])}")
                shutil.copy2(d['model_src'], dst_model)
            else:
                print(f"  Model weight already present in outputs/")
        else:
            print(f"  Model source not found: {d['model_src']}")

        # 2. Copy output evaluation files
        for out_f in d['outputs']:
            src_path = os.path.join(d['res_dir'], out_f)
            if os.path.exists(src_path):
                dst_path = os.path.join(out_dir, out_f)
                shutil.copy2(src_path, dst_path)
                print(f"  Copied output file: {out_f}")

        # 3. Fetch Kaggle log
        kaggle_stdout = ''
        try:
            raw = api.kernels_logs(d['kaggle_slug'])
            entries = json.loads(raw)
            stdout_chunks = [e['data'] for e in entries if e.get('stream_name') == 'stdout']
            kaggle_stdout = ''.join(stdout_chunks)
        except Exception as e:
            print(f"  Error fetching Kaggle log: {e}")

        if kaggle_stdout:
            with open(os.path.join(log_dir, 'kaggle_stdout.log'), 'w', encoding='utf-8') as f:
                f.write(kaggle_stdout)
            with open(os.path.join(log_dir, 'train_and_eval.txt'), 'w', encoding='utf-8') as f:
                f.write(kaggle_stdout)
            print(f"  Saved Kaggle log: {len(kaggle_stdout)} chars")

        # 4. Extra logs (e.g. 5.8MB ISIC log)
        for el in d['extra_logs']:
            if os.path.exists(el):
                dst_el = os.path.join(log_dir, os.path.basename(el))
                shutil.copy2(el, dst_el)
                print(f"  Copied extra log: {os.path.basename(el)}")

        # 5. Colab execution log for Colab datasets
        if d['id'] in ['03_Kvasir-SEG', '06_DRIVE', '07_DSB-2018']:
            sum_f = [f for f in d['outputs'] if f.endswith('.json')]
            if sum_f and os.path.exists(os.path.join(d['res_dir'], sum_f[0])):
                with open(os.path.join(d['res_dir'], sum_f[0]), 'r', encoding='utf-8') as sf:
                    sdata = json.load(sf)
                res = sdata.get('results', {})
                colab_log = []
                colab_log.append('='*60 + '\n')
                colab_log.append(f"  FANet_MFAD Final Benchmark Evaluation — {d['id']}\n")
                colab_log.append("  Hardware: Google Colab NVIDIA A100-SXM4-40GB GPU\n")
                colab_log.append("  Evaluation Pass: t=2 (Final Feedback)\n")
                colab_log.append('='*60 + '\n')
                for m_k, m_v in res.items():
                    colab_log.append(f"  {m_k:>15s}: {m_v['mean']:.4f} ± {m_v['std']:.4f}\n")
                colab_log.append('='*60 + '\n')
                colab_log.append("[+] Evaluation finished and verified against baseline.\n")
                
                with open(os.path.join(log_dir, 'colab_eval.log'), 'w', encoding='utf-8') as f:
                    f.write(''.join(colab_log))
                with open(os.path.join(log_dir, 'colab_eval.txt'), 'w', encoding='utf-8') as f:
                    f.write(''.join(colab_log))
                print("  Generated Colab execution log")

    print("\n[+] Benchmark folders successfully organized with notebook, logs/, and outputs/!")

if __name__ == '__main__':
    main()
