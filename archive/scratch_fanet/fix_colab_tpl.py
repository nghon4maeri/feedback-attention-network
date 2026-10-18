p = 'scripts/make_colab_notebooks.py'
s = open(p, encoding='utf-8').read()
s = s.replace('mkdir -p {DATA_ROOT}', 'mkdir -p {{DATA_ROOT}}').replace('-p {DATA_ROOT} --unzip', '-p {{DATA_ROOT}} --unzip')
s = s.replace('DATA_ROOT = "/content/data/{slug}"        # local SSD = fast IO (data is NOT kept on Drive)',
              'DATA_ROOT = "/content/{slug}"        # local SSD = fast IO (data is NOT kept on Drive)')
s = s.replace('(recommended) or upload `kaggle.json` when asked.', '(left sidebar, key icon, enable notebook access).')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
