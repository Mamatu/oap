/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_NEURAL_TESTS__DATA_MISCS_H
#define OAP_NEURAL_TESTS__DATA_MISCS_H

#include "oapNeuralTests_Types.h"

namespace oap
{
namespace Backpropagation_Data_Miscs
{
namespace Test_1
{

IdxsToCheck g_idxsToCheck =
{
  {0, 1, 2, 3, 4, 5, 6, 7, 8},
  {0, 1, 2, 3}
};

std::vector<Weights> g_weights1to2Vec =
{
  {
    0.2,
    0.2,
    0.1,
    0.2,
    0.2,
    0.1,
    0.2,
    0.2,
    0.1,
  },
  {
    0.2015415656559217,
    0.2021262554160332,
    0.1038135365011816,
    0.2015415656559217,
    0.2021262554160332,
    0.1038135365011816,
    0.2015415656559217,
    0.2021262554160332,
    0.1038135365011816,
  }
};

std::vector<Weights> g_weights2to3Vec =
{
  {
    0.2,
    0.2,
    0.2,
    0.1,
  },
  {
    0.20569433279687238,
    0.20569433279687238,
    0.20569433279687238,
    0.12068242867556898,
  }
};

Batches g_batch =
{
  {
    {{0.44357233490399445, 0.22756905427903037}, 1},
    {{0.3580909454680603, 0.8306780543693363}, 1},
  }
};

}

namespace Test_2
{

IdxsToCheck g_idxsToCheck =
{
  {0, 1, 2, 3, 4, 5, 6, 7, 8},
  {0, 1, 2, 3}
};

std::vector<Weights> g_weights1to2Vec =
{
  {
    0.18889833379294582,
    0.18741691262115692,
    0.14100651782253998,
    0.18889833379294582,
    0.18741691262115692,
    0.14100651782253998,
    0.18889833379294582,
    0.18741691262115692,
    0.14100651782253998,
  },
  {
    0.18747739212831752, 
    0.18711107569519983, 
    0.1397053002260854, 
    0.18747739212831752, 
    0.18711107569519983, 
    0.1397053002260854, 
    0.18747739212831752,
    0.18711107569519983,
    0.1397053002260854,
  }
};

std::vector<Weights> g_weights2to3Vec =
{
  {
    0.1243037373648706,
    0.1243037373648706,
    0.1243037373648706,
    0.05120154361408274, 
  },
  {
    0.11308197717424648, 
    0.11308197717424648,
    0.11308197717424648,
    0.028814535268009908
  }
};

Batches g_batch =
{
  {
    {{1.1171665268436015, 1.6264896739229502}, 1},
    {{1.9827643776881154, 3.1666823397044954}, -1},
    {{-3.7939263802800536, 0.6280114688227496}, -1},
    {{3.1655171307757155, 3.690154247154129}, -1},
    {{4.3098981190509935, -1.8380685678345827}, -1},
  }
};

}

namespace Test_3
{

IdxsToCheck g_idxsToCheck =
{
  {0, 1, 2, 3, 4, 5, 6, 7, 8},
  {0, 1, 2, 3}
};

std::vector<Weights> g_weights1to2Vec =
{
  {
    0.2,
    0.2,
    0.1,
    0.2,
    0.2,
    0.1,
    0.2,
    0.2,
    0.1,
  },
  {
    0.19908199840345,
    0.19356847603468466,
    0.10580908512486277,
    0.19908199840345,
    0.19356847603468466,
    0.10580908512486277,
    0.19908199840345,
    0.19356847603468466,
    0.10580908512486277,   
  }
};

std::vector<Weights> g_weights2to3Vec =
{
  {
    0.2,
    0.2,
    0.2,
    0.1,
  },
  {
    0.1954852905493779,
    0.1954852905493779,
    0.1954852905493779,
    0.1297309930846901,
  }

};

Batches g_batch =
{
  {
    {{-0.15802860120278975, -1.1071492028561536}, 1},
  }
};

}

namespace Test_4
{

IdxsToCheck g_idxsToCheck =
{
  {0, 1, 2, 3, 4, 5, 6, 7, 8},
  {0, 1, 2, 3}
};

namespace
{
Weights w1to2step0 = {0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0, 0, 0};
Weights w1to2step1 =
{
  0.20182628407433365,
  0.20093695144385168,
  0.10411721816404285,
  0.20182628407433365,
  0.20093695144385168,
  0.10411721816404285,
  0.20182628407433365,
  0.20093695144385168,
  0.10411721816404285,
};

Weights w2to3step0 = { 0.2, 0.2, 0.2, 0.1};
Weights w2to3step1 =
{
  0.20500015007570618,
  0.20500015007570618,
  0.20500015007570618,
  0.12173630913135866
};
}

std::vector<Weights> g_weights1to2Vec =
{
  w1to2step0, w1to2step1
};


std::vector<Weights> g_weights2to3Vec =
{
  w2to3step0, w2to3step1
};

Batches g_batch =
{
  {
    {{0.44357233490399445, 0.22756905427903037}, 1},
  }
};

}

}
}
#endif
