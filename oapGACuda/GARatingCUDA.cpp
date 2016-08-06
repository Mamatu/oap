/* 
 * File:   GAComponentCPU.cpp
 * Author: mmatula
 * 
 * Created on July 7, 2013, 5:50 PM
 */

#include "GARatingCUDA.h"
namespace ga {

    GARatingInstanceCUDA::GARatingInstanceCUDA(GAProcessCUDA* gaProcessCUDA) :
    ga::GAExecutableConfigurator((core::ProcessControler*)gaProcessCUDA, true, false, false, false) {
    }

    GARatingCUDA::GARatingCUDA(const char* name) : ga::GAModuleComponent(name) {
    }

    GARatingCUDA::~GARatingCUDA() {
    }

    ga::GAExecutableConfigurator* GARatingCUDA::newGAExtensionInstance() {
        return this->newGARatingInstanceCUDA();
    }

    void GARatingCUDA::deleteGAExtensionInstance(ga::GAExecutableConfigurator* object) {
        GARatingInstanceCUDA* ptr = NULL;
        if ((ptr = dynamic_cast<ga::GARatingInstanceCUDA*> (object)) != NULL) {
            this->deleteGARatingInstanceCUDA(ptr);
        }
    }

    ga::GAExecutable* GARatingInstanceCUDA::newGAExecutableExtension() {
        return this->newGARatingExecutableCUDA();
    }

    void GARatingInstanceCUDA::deleteGAExecutableExtension(ga::GAExecutable* object) {
        GARatingExecutorCUDA* ptr = dynamic_cast<GARatingExecutorCUDA*> (object);
        if (ptr) {
            this->deleteGARatingExecutableCUDA(ptr);
        }
    }
}