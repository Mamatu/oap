
#include "PythonApi.h"

#ifdef __cplusplus
extern "C" {
#endif
      
    static PyModuleDef oai_server_module = {
        PyModuleDef_HEAD_INIT, "oai", NULL, -1, NULL
    };

    PyMODINIT_FUNC PyInit_oai() {
        Py_InitModule();
        return PyModule_Create(&oai_server_module);
    }

#ifdef __cplusplus
}
#endif