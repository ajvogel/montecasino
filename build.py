from setuptools import Extension, Distribution
import shutil, os
from pathlib import Path
from Cython.Build import cythonize
from Cython.Distutils.build_ext import build_ext as cython_build_ext



def build(setup_kwds):
    """

    https://github.com/aotuai/example-cython-poetry-pypi/blob/main/build.py
    https://github.com/python-poetry/poetry/issues/2740
    """

    # extMod = Extension(
    #     name="casino.core",
    #     sources=["casino/core.py"]
    # )

    cythonized = cythonize([
        "casino/core.py",
        "casino/random.py"
                           ],
        annotate=True,
        force=True,
    include_path=['casino/'])

    dist = Distribution({
        "ext_modules":cythonized,
        "cmdclass": {
            "build_ext": cython_build_ext,
        },        
    })


    dist.run_command("build_ext")
    cmd = dist.get_command_obj("build_ext")


    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)
    
    # breakpoint()          
    # build_ext_cmd.copy_extensions_to_source() 
    
if __name__ == "__main__":
    build(None)


# fib_source = Path('casino/core.py')

# # distutils magic. This is essentially the same as calling
# # python setup.py build_ext --inplace
# dist = Distribution(attrs={'ext_modules': cythonize(fib_source.name)})
# build_ext_cmd = dist.get_command_obj('build_ext')
# build_ext_cmd.ensure_finalized()
# build_ext_cmd.inplace = 1
# build_ext_cmd.run()

# fib_obj = Path(build_ext_cmd.get_ext_fullpath(fib_source.stem))

# the lib was built, so the import will succeed now
# from fib import fib


# def teardown_module():
#     # remove built library
#     fib_obj.unlink()

#     # if you also want to clean the build dir:
#     from distutils.dir_util import remove_tree
#     remove_tree(build_ext_cmd.build_lib)
#     remove_tree(build_ext_cmd.build_temp)


# # sample tests

# def test_zero():
#     assert fib(0) == 0


# def test_ten():
#     assert fib(10) == 55



# import os
# import shutil
# from pathlib import Path

# # Uncommend if your library can still function if extensions fail to compile.
# allowed_to_fail = False
# # allowed_to_fail = os.environ.get("CIBUILDWHEEL", "0") != "1"


# def build_cython_extensions():
#     # when using setuptools, you should import setuptools before Cython,
#     # otherwise, both might disagree about the class to use.
#     from setuptools import Extension  # noqa: I001
#     from setuptools.dist import Distribution  # noqa: I001
#     import Cython.Compiler.Options  # pyright: ignore [reportMissingImports]
#     from Cython.Build import build_ext, cythonize  # pyright: ignore [reportMissingImports]

#     Cython.Compiler.Options.annotate = True

#     if os.name == "nt":  # Windows
#         extra_compile_args = [
#             "/O2",
#         ]
#     else:  # UNIX-based systems
#         extra_compile_args = [
#             "-O3",
#             "-Werror",
#             "-Wno-unreachable-code-fallthrough",
#             "-Wno-deprecated-declarations",
#             "-Wno-parentheses-equality",
#         ]
#     # Relative to project root director
#     include_dirs = [
#         "pythontemplate/",
#         "pythontemplate/_c_src",
#     ]

#     c_files = [str(x) for x in Path("pythontemplate/_c_src").rglob("*.c")]
#     extensions = [
#         Extension(
#             # Your .pyx file will be available to cpython at this location.
#             "pythontemplate._c_extension",
#             [
#                 # ".c" and ".pyx" source file paths
#                 "pythontemplate/_c_extension.pyx",
#                 *c_files,
#             ],
#             include_dirs=include_dirs,
#             extra_compile_args=extra_compile_args,
#             language="c",
#         ),
#     ]

#     include_dirs = set()
#     for extension in extensions:
#         include_dirs.update(extension.include_dirs)
#     include_dirs = list(include_dirs)

#     ext_modules = cythonize(extensions, include_path=include_dirs, language_level=3, annotate=True)
#     dist = Distribution({"ext_modules": ext_modules})
#     cmd = build_ext(dist)
#     cmd.ensure_finalized()
#     cmd.run()

#     for output in cmd.get_outputs():
#         output = Path(output)
#         relative_extension = output.relative_to(cmd.build_lib)
#         shutil.copyfile(output, relative_extension)




# dfdfd


# try:
#     build_cython_extensions()
# except Exception:
#     if not allowed_to_fail:
#         raise


