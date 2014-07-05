/* 
 * File:   GATypes.h
 * Author: mmatula
 *
 * Created on July 3, 2013, 8:06 PM
 */

#ifndef OGLA_GA_TYPES_H
#define	OGLA_GA_TYPES_H

#include <string.h>
#include "Types.h"
#include "Math.h"
#include "WrapperInterfaces.h"

namespace ga {

    class GAData {
    private:
        bool onLocal;
    protected:
        virtual ~GAData();
    public:

        class Generation {
        private:
            GAData* gaData;
            friend class GAData;
        protected:
            virtual ~Generation();
        public:

            GAData* getGAData() const {
                return gaData;
            }

            Generation();
            /**
             * Sizes of genes. Size of gene must be equal or lower than ::realGeneSize. 
             */
            uintt* genesSizes;

            /**
             * Size of genomes. Size is equals to number of genomes which are contained in genome. 
             */
            uintt* genomesSizes;

            /**
             * Genome - array of gnumbers which represent genetic data of this generation (genomes0.
             */
            floatt* genomes;
        };

        uintt generationCounter;

        /**
         * Number of genomes of the generation .
         */
        uintt genomesCount;

        /**
         * 2 * genomesCount
         */
        uintt* selectedGenomes;

        /**
         * Real size of gene (number of units which are contained in gene). 
         */
        uintt realGeneSize;

        /**
         * Real size of genome (number of genes which are contained in genome).
         */
        uintt realGenomeSize;

        floatt * ranks;


        Generation previousGeneration;
        Generation currentGeneration;
        GAData();
    };
}

#endif	/* GATYPES_H */

