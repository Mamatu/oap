/* 
 * File:   GAComponentCPU.h
 * Author: mmatula
 *
 * Created on July 7, 2013, 5:50 PM
 */

#ifndef GACOMPONENTCPU_H
#define	GACOMPONENTCPU_H

#include "threads.h"
#include "GACore.h"
#include "Functions.h"
#include "Process.h"
#include <gsl/gsl_rng.h>
#include "GAExtension.h"

namespace ga {

    class GARatingExecutorCUDA : public ga::GAExecutableRawImpl {
    public:
        virtual void setData(void* dataPtr) = 0;
    };

    class GARatingCUDA;

    class GARatingInstanceCUDA : public ga::GAExecutableConfigurator {
    protected:
        GARatingInstanceCUDA(GAProcessCUDA* gaProcessCUDA);
        virtual ~GARatingInstanceCUDA();

        ga::GAExecutable* newGAExecutableExtension();
        void deleteGAExecutableExtension(ga::GAExecutable* object);

        virtual GARatingExecutorCUDA* newGARatingExecutableCUDA() = 0;
        virtual void deleteGARatingExecutableCUDA(GARatingExecutorCUDA*) = 0;
    };

    class GARatingCUDA : public ga::GAModuleComponent {
    protected:
        GARatingCUDA(const char* name);
        virtual ~GARatingCUDA();

        ga::GAExecutableConfigurator* newGAExtensionInstance();
        void deleteGAExtensionInstance(ga::GAExecutableConfigurator* object);

        virtual GARatingInstanceCUDA* newGARatingInstanceCUDA() = 0;
        virtual void deleteGARatingInstanceCUDA(GARatingInstanceCUDA* object) = 0;
    };
}
#endif	/* GACOMPONENTCPU_H */

