conda create --prefix .\envs  
conda activate -p .\envs\  
pip install -r .\requirements.txt       -> before that changed "sklearn" in the requirements.txt file to "scikit-learn"  
conda config --set env_prompt '({name})'    -> to show (envs) instead of whole path in terminal

- downloaded SMPL_python_v.1.1.0 from their official website  -> https://smpl.is.tue.mpg.de/index.html
- moved "smpl" folder to "bodies-at-rest" (main dir)
- to run the .sh files I have to open terminal in VSCode as ubuntu (WSL) instead of default powershell
- also in the .sh files I have to add "/" at the end of paths of mkdir
    for eg: mkdir -p ../data_BR/synth/quick_test/

*TO RUN viz_synth_cvpr_release.py*

- SyntaxError: Missing parentheses in call to 'print'. Did you mean print(...)?
  added paranthesis around print statement in the files:
    - viz_synth_cvpr_release.py
        - line 218
        - line 219
    - lib_pyrender_basic.py
        - line 11

- replaced import cPickle as pkl -> with import pickle as pkl
    - viz_synth_cvpr_release.py
        - line 27
        - line 44
    
    - smpl\smpl_webuser\serialization.py
        - line 26

- replaced import cPickle as pickle -> with import pickle as pickle
    - lib_pyrender_basic.py
        - line 29

- to install open3d library
    - donwgraded python from 3.12 to 3.9 using -> conda install python=3.9
    - pip install addict
    conda install -c open3d-admin open3d

- ImportError: cannot import name '_imaging' from 'PIL'
    - pip uninstall pillow
    - pip install pillow

- [Warning] Since Open3D 0.15, installing Open3D via conda is deprecated. Please re-install Open3D via: `pip install open3d -U`.
    installed

- Try downgrading networkx to a previous version that doesn't have this issue. For example, you can try downgrading to version 2.5:
    pip install networkx==2.5

- pip uninstall matplotlib
    pip install matplotlib

- pip uninstall kiwisolver
    pip install kiwisolver

- pip uninstall torch
    pip install torch



- ModuleNotFoundError: No module named 'smpl'
    instead of running like -> d:/Coding/projects_git_repos/bodies-at-rest/envs/python.exe d:/Coding/projects_git_repos/bodies-at-rest/PressurePose/viz_synth_cvpr_release.py

    run it from where the script is, i.e. cd PressurePose
    then run -> viz_synth_cvpr_release.py
    
- pip install numpy==1.19.5 to downgrade 1.26.4 upgraded again to 1.23

- from posemapper import posemap
  ModuleNotFoundError: No module named 'posemapper'
    $env:PYTHONPATH += ";D:\Coding\projects_git_repos\bodies-at-rest\smpl\smpl_webuser\"

- print "" instead of print("")
    replaced using ctrl shift F but still
    this is python 2 syntax
    uninstalling env and creating new env with python 2.7