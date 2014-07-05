/* 
 * File:   PythonApi.h
 * Author: mmatula
 *
 * Created on September 11, 2013, 11:38 PM
 */

#ifndef PYTHONAPI_H
#define	PYTHONAPI_H


#include <python3.2/Python.h>
#include "APIServer.h"
#include "APIClient.h"

typedef struct {
    PyObject_HEAD
    core::APIServer* apiServer;
} ApiServerObject;

typedef struct {
    PyObject_HEAD
    core::APIInterface* apiClient;
} ApiClientObject;

void Py_InitModule();
void Py_DestroyModule();

#endif	/* PYTHONAPI_H */

