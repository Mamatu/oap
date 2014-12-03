from distutils.core import setup, Extension

mode = "Debug"
core_path = "/home/mmatula/wProject"

def get_paths(projects_names):
    global mode
    global core_path
    out = ['%s/%s'%(core_path, fa) for fa in projects_names]
    return out

def get_paths_to_dirs(projects_names):
    global mode
    global core_path
    out = ['%s/%s/dist/%s/GNU-Linux-x86/'%(core_path, fa, mode) for fa in projects_names]
    return out

def get_libraries(projects_names, ext = None):
    global mode
    global core_path
    out = ["%s"%(fa) for fa in projects_names]
    if ext != None:
        out1 = ["%s"%(fa) for fa in ext]
        out = out + out1
    return out

projects = ['wClient','wServer','wUtils']
ext = ['boost_python']

module1 = Extension("oai",library_dirs=get_paths_to_dirs(projects),libraries=get_libraries(projects,ext),
sources=["main.cpp","PythonApi.cpp"], include_dirs=get_paths(projects))

setup(
    name = 'oai',
    ext_modules = [module1]
)

