from pythonforandroid.recipe import PythonRecipe
from pythonforandroid.toolchain import shprint, current_directory
from os.path import join
import sh


class MediaPipeRecipe(PythonRecipe):
    version = '0.8.7.3'  # replace this with the version of MediaPipe you want
    url = 'https://github.com/google/mediapipe/archive/refs/tags/v{version}.tar.gz'

    depends = ['protobuf_cpp', 'opencv', 'python3']

    patches = ['path_to_your_patches_if_required']

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['PYTHONPATH'] = ':'.join([
            join(self.get_build_dir(arch.arch), 'python'),
            env['PYTHONPATH'],
        ])

        env['USE_SYSTEM_PROTOBUF'] = '0'
        return env

    def build_arch(self, arch):
        super().build_arch(arch)
        with current_directory(self.get_build_dir(arch.arch)):
            env = self.get_recipe_env(arch)
            # Make sure you have bazel installed and configured
            shprint(sh.bazel, 'build', '-c', 'opt', 
                    '--define', 'MEDIAPIPE_DISABLE_GPU=1', 
                    '--action_env', 'PYTHON_BIN_PATH=/usr/bin/python3',
                    'mediapipe/python:_framework_bindings.so',
                    _env=env)


recipe = MediaPipeRecipe()
