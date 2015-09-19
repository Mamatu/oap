#include "StackAnalyzer.h"

namespace stack {
inline llu* getPrevRBPAddress(llu* rbp) {
  llu value_rbp = *rbp;
  return reinterpret_cast<llu*>(value_rbp);
}

inline llu* getPRBPA(llu* rbp) { return getPrevRBPAddress(rbp); }

void getInstructionsPointers(std::vector<uintt>& rips,
                             std::vector<uintt>& rbps) {
  register llu* RBP asm("rbp");
  register llu* RSP asm("rsp");
  llu* tempRBP = RBP;
  do {
    fprintf(stderr, "pointer = %x \n", *tempRBP);
    fprintf(stderr, "pointer = %x \n", *(tempRBP + 1));
    rbps.push_back(*tempRBP);
    rips.push_back(*(tempRBP + 1));
    tempRBP = getPRBPA(tempRBP);
  } while (tempRBP != NULL);
}
}
