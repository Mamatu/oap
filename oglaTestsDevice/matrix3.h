#ifndef MATRIX3
#define MATRIX3

const char* matrix3Str =
    "  (columns=32, rows=32) [-12.5968, -4.06124, -0.000266417, -2.46986e-06, "
    "-1.64278e-05, -1.4357e-06, 4.74251e-06, 2.03352e-07, -9.57578e-08, "
    "9.4242e-09, 3.86994e-09, 3.1275e-11, 1.8469e-12, -1.61326e-14, "
    "-3.40264e-15, -1.87296e-14, 4.42018e-15, -9.71292e-15, -5.64341e-16, "
    "2.26109e-14, 5.78878e-15, -6.93526e-15, 7.69117e-15, -4.25258e-15, "
    "6.55684e-16, 6.18975e-15, 3.72411e-15, 7.51853e-15, 1.04477e-14, "
    "-1.07504e-14, 9.4756e-15, 6.21372e-12 | -4.06124, 12.5968, 0.000841458, "
    "7.80087e-06, 5.18859e-05, 4.53454e-06, -1.49789e-05, -6.42272e-07, "
    "3.02444e-07, -2.97656e-08, -1.2223e-08, -9.87887e-11, -5.83378e-12, "
    "4.41695e-14, -2.62552e-14, 1.60844e-15, 6.5474e-15, -7.11891e-15, "
    "1.68735e-14, 1.83189e-14, 1.85462e-14, -6.31063e-15, 1.44149e-14, "
    "-8.0827e-15, -3.65431e-16, 6.10913e-15, 1.05387e-14, 1.3836e-14, "
    "1.28774e-14, -1.5715e-14, 9.85169e-15, 1.3689e-11 | -0.000266417, "
    "0.000841458, -11.6797, -0.000999743, 3.87815e-06, 2.56558e-09, "
    "1.3561e-11, -1.77379e-10, 3.17875e-12, -1.82662e-15, -1.8622e-15, "
    "-1.06567e-13, -2.55245e-13, -1.85395e-14, -5.7156e-15, 2.78876e-15, "
    "2.88093e-15, 6.51505e-15, 5.84202e-15, -1.02254e-14, 7.60978e-15, "
    "3.74937e-15, 5.15256e-15, -4.20915e-15, -2.40119e-15, -1.4764e-15, "
    "4.05785e-15, 8.8693e-15, 2.64627e-15, 6.28084e-14, -8.67146e-12, "
    "5.09877e-08 | -2.46986e-06, 7.80087e-06, -0.000999743, 11.6626, "
    "-0.000993616, -6.57327e-07, 2.99387e-09, -3.54654e-10, -1.98191e-10, "
    "4.64398e-14, 5.25482e-14, -5.34284e-13, -1.38673e-12, -1.36876e-13, "
    "-7.59276e-15, 8.5894e-15, -3.54196e-15, -4.59892e-14, 5.60185e-15, "
    "2.16022e-15, -1.47052e-14, 1.01805e-14, 7.45975e-15, 9.62733e-15, "
    "4.22918e-15, -5.60894e-17, -8.91131e-15, 1.46826e-15, 5.20315e-15, "
    "7.22994e-13, -8.96637e-11, 4.55672e-07 | -1.64278e-05, 5.18859e-05, "
    "3.87815e-06, -0.000993616, -11.5279, 0.000997105, -1.00165e-06, "
    "4.33366e-07, -1.93156e-09, 3.77309e-13, -2.14337e-13, 1.06641e-11, "
    "2.60668e-11, 1.70889e-12, -9.15038e-15, 1.60151e-14, -1.8717e-15, "
    "-2.22452e-13, -1.91634e-14, 8.22567e-15, -5.35718e-15, 1.7611e-14, "
    "7.50035e-15, 5.11447e-15, 9.58606e-15, 1.1102e-14, 8.6866e-16, "
    "-2.36895e-15, 3.97291e-15, -1.32096e-12, 1.65636e-10, -9.00336e-07 | "
    "-1.4357e-06, 4.53454e-06, 2.56559e-09, -6.57327e-07, 0.000997105, "
    "11.4267, -0.000998296, 7.61434e-07, 7.04471e-08, -1.38516e-11, "
    "-7.60723e-12, 1.23034e-11, 3.0138e-11, 1.58019e-12, -1.26282e-15, "
    "1.0011e-15, -1.80041e-15, 1.96769e-13, 1.34495e-14, 1.32206e-15, "
    "2.95319e-15, -4.0931e-15, -1.44119e-14, -9.59281e-15, -2.78573e-15, "
    "-3.3165e-15, 1.19206e-15, -2.829e-15, 6.11602e-16, -1.20565e-12, "
    "1.50008e-10, -6.7344e-07 | 4.74251e-06, -1.49789e-05, 1.35633e-11, "
    "2.99386e-09, -1.00165e-06, -0.000998296, -11.2807, 0.000977993, "
    "-4.20946e-07, 8.26463e-11, -1.6734e-11, 1.39194e-10, 3.40735e-10, "
    "1.31097e-11, -4.20754e-14, 4.59051e-14, -4.43721e-17, -6.17238e-13, "
    "-6.30257e-14, 1.27702e-15, -2.50347e-16, 1.74342e-14, -3.86802e-16, "
    "2.59182e-15, 4.40118e-15, -2.19849e-15, -6.8187e-15, 2.49043e-15, "
    "-2.41668e-15, -1.11876e-12, 1.39677e-10, -6.67687e-07 | 2.03352e-07, "
    "-6.42272e-07, -1.7739e-10, -3.54654e-10, 4.33366e-07, 7.61434e-07, "
    "0.000977993, 11.0257, 0.000981707, -1.92728e-07, -2.12737e-08, "
    "1.7644e-09, 4.31918e-09, 9.41571e-11, -3.23831e-13, 3.27416e-13, "
    "-3.6787e-15, 2.14139e-12, 2.17293e-13, -2.14139e-15, 1.0145e-14, "
    "9.16984e-15, -8.52006e-17, -2.79783e-15, 2.88434e-16, -4.9305e-15, "
    "2.33621e-15, -1.4785e-15, 3.21029e-15, -2.15023e-12, 2.69054e-10, "
    "-9.72766e-07 | -9.57578e-08, 3.02444e-07, 3.21393e-12, -1.98192e-10, "
    "-1.93155e-09, 7.04471e-08, -4.20946e-07, 0.000981707, 10.7681, "
    "-0.000970587, -1.50839e-06, 1.19229e-08, 2.9187e-08, 3.71826e-10, "
    "-1.23044e-12, 1.25759e-12, -7.52664e-15, 2.94441e-12, 2.84827e-13, "
    "7.42119e-15, -4.73189e-15, 4.13476e-15, -6.61608e-15, 3.69898e-15, "
    "-7.05304e-15, 6.4881e-15, -1.96678e-16, -1.46317e-16, 6.90893e-15, "
    "-8.87666e-13, 1.10465e-10, -3.46431e-07 | 9.4242e-09, -2.97656e-08, "
    "-6.25962e-16, 3.8887e-14, 3.79067e-13, -1.383e-11, 8.26396e-11, "
    "-1.92728e-07, -0.000970587, -10.3875, 0.000995881, -1.9539e-05, "
    "-4.7831e-05, -2.75818e-07, 9.28047e-10, -9.38175e-10, -7.13922e-13, "
    "2.75367e-10, 2.74085e-11, -2.27315e-13, 6.91264e-14, -1.83565e-15, "
    "1.53316e-15, 3.52935e-15, 3.9004e-15, 4.20496e-16, 1.06424e-15, "
    "-3.04197e-15, 3.89048e-15, 7.09001e-12, -8.83758e-10, 2.59296e-06 | "
    "3.86994e-09, -1.22229e-08, 7.32522e-16, 5.9006e-14, -2.12912e-13, "
    "-7.62556e-12, -1.67357e-11, -2.12737e-08, -1.50839e-06, 0.000995881, "
    "-10.0517, 0.000373807, 0.000915068, 2.6507e-06, -8.91882e-09, "
    "9.01626e-09, 6.80121e-12, -6.60266e-10, -6.57142e-11, 5.24618e-13, "
    "-1.77281e-13, -1.39173e-15, -8.40683e-15, 5.55511e-15, 9.69376e-15, "
    "-1.41137e-14, 1.96456e-15, -2.73732e-15, -7.7644e-15, -3.22731e-12, "
    "4.0192e-10, -9.72513e-07 | 3.12733e-11, -9.87741e-11, -1.05611e-13, "
    "-5.68018e-13, 1.06602e-11, 1.231e-11, 1.39198e-10, 1.76439e-09, "
    "1.19229e-08, -1.9539e-05, 0.000373807, -9.57774, 0.000999538, "
    "-0.000933749, 3.14179e-06, -3.17611e-06, -2.3966e-09, 2.27189e-08, "
    "2.26127e-09, -1.83288e-11, 5.98249e-12, 2.57238e-15, -3.13972e-15, "
    "2.14153e-15, 3.32556e-15, -5.47706e-15, 1.28857e-15, 4.82587e-15, "
    "8.9329e-15, 9.30478e-12, -1.15901e-09, 2.10394e-06 | 1.84634e-12, "
    "-5.83158e-12, -2.58555e-13, -1.39024e-12, 2.60961e-11, 3.01349e-11, "
    "3.40752e-10, 4.31917e-09, 2.9187e-08, -4.7831e-05, 0.000915068, "
    "0.000999538, 9.55362, 0.000351861, -1.18391e-06, 1.19684e-06, "
    "9.03093e-10, 1.74292e-08, 1.73477e-09, -1.40665e-11, 4.58046e-12, "
    "1.97431e-15, 8.71938e-15, 5.23835e-15, 5.05932e-15, 4.29575e-15, "
    "-2.64057e-15, 1.02298e-14, 7.08707e-15, -1.32314e-11, 1.65081e-09, "
    "-2.50047e-06 | -1.31327e-14, 4.15061e-14, -2.41835e-14, -1.23314e-13, "
    "1.73847e-12, 1.58085e-12, 1.31047e-11, 9.41583e-11, 3.71828e-10, "
    "-2.75818e-07, 2.6507e-06, -0.000933749, 0.000351861, 8.85346, "
    "0.000973221, -1.65154e-05, -1.78508e-07, 1.22152e-06, 1.21581e-07, "
    "-9.85337e-10, 3.21308e-10, -5.43937e-15, -3.22793e-15, 4.28189e-15, "
    "-1.79836e-15, -1.92352e-15, 8.27846e-15, 7.36792e-14, 3.11955e-14, "
    "-2.26041e-11, 2.81415e-09, -2.68317e-06 | 3.0815e-15, -9.30037e-15, "
    "-8.64071e-18, 2.75476e-16, -5.80398e-15, -5.6116e-15, -4.42364e-14, "
    "-3.16743e-13, -1.2509e-12, 9.28048e-10, -8.91883e-09, 3.14179e-06, "
    "-1.18391e-06, 0.000973221, -8.41053, -0.000996058, -2.0612e-05, "
    "-1.96033e-05, -1.95117e-06, 1.58131e-08, -5.15641e-09, 3.58044e-15, "
    "7.55755e-15, 6.50301e-16, 3.69039e-15, 4.70526e-15, 1.69869e-15, "
    "-4.23747e-14, -3.16105e-14, -1.35325e-11, 1.68022e-09, -1.4163e-06 | "
    "-3.49059e-16, 9.84914e-16, -2.38716e-16, -3.17683e-16, 5.83324e-15, "
    "5.09784e-15, 4.45389e-14, 3.20274e-13, 1.26474e-12, -9.38185e-10, "
    "9.01624e-09, -3.17611e-06, 1.19684e-06, -1.65154e-05, -0.000996058, "
    "8.35473, 0.000968965, -1.37922e-05, -1.37278e-06, 1.11255e-08, "
    "-3.62786e-09, -6.25307e-15, 4.00722e-16, 1.18955e-14, 9.08247e-15, "
    "4.04184e-15, 1.65437e-14, -1.04452e-13, -2.83539e-14, 1.745e-11, "
    "-2.17121e-09, 1.45467e-06 | -8.82834e-17, -9.11273e-18, 6.31432e-17, "
    "-2.05362e-16, -1.02642e-17, 3.77021e-17, 1.69304e-16, 3.79481e-16, "
    "9.02285e-16, -7.07793e-13, 6.80333e-12, -2.3966e-09, 9.03102e-10, "
    "-1.78508e-07, -2.0612e-05, 0.000968965, -7.92412, -0.000947201, "
    "-9.42774e-05, 7.64062e-07, -2.49149e-07, -6.11381e-16, 4.47929e-16, "
    "-1.66536e-14, 5.47753e-14, 4.09771e-15, -7.27524e-15, -3.25434e-13, "
    "-2.11799e-13, -4.15049e-11, 5.13737e-09, -3.04939e-06 | -3.16974e-17, "
    "-4.0404e-17, 5.97217e-15, -3.95669e-14, -2.31118e-13, 1.90553e-13, "
    "-6.06507e-13, 2.14282e-12, 2.95371e-12, 2.75373e-10, -6.60277e-10, "
    "2.27189e-08, 1.74292e-08, 1.22152e-06, -1.96033e-05, -1.37922e-05, "
    "-0.000947201, 6.94774, -0.000987185, 3.52556e-05, 1.26824e-05, "
    "2.69944e-07, -2.79311e-10, -2.39755e-11, 3.54571e-11, 1.53635e-13, "
    "5.96115e-15, 1.82028e-11, 4.62117e-12, -2.39849e-10, 2.94329e-08, "
    "-6.40908e-06 | 1.60493e-16, 1.2158e-16, 5.16126e-16, -3.74735e-15, "
    "-2.32284e-14, 1.94289e-14, -6.03657e-14, 2.12738e-13, 2.94029e-13, "
    "2.74081e-11, -6.57191e-11, 2.26127e-09, 1.73478e-09, 1.21581e-07, "
    "-1.95117e-06, -1.37278e-06, -9.42774e-05, -0.000987185, -6.67729, "
    "0.000880533, 0.000316752, 3.38101e-06, -3.49826e-09, -1.33992e-10, "
    "1.98226e-10, -7.62501e-13, 3.94146e-16, -1.06796e-11, -5.90079e-12, "
    "-1.5165e-10, 1.82773e-08, -3.96655e-06 | 3.88533e-17, 4.99204e-17, "
    "2.5337e-16, 1.05778e-16, 9.36291e-17, -2.99206e-16, 2.19017e-16, "
    "-1.77946e-15, -2.4226e-15, -2.22312e-13, 5.32581e-13, -1.83263e-11, "
    "-1.40594e-11, -9.85344e-10, 1.58131e-08, 1.11255e-08, 7.64062e-07, "
    "3.52556e-05, 0.000880533, 6.08774, 0.739324, 0.000317987, -3.29015e-07, "
    "-7.06788e-09, 1.04568e-08, 2.45081e-11, 5.58672e-14, 2.09369e-10, "
    "4.18447e-11, -5.92705e-10, 7.06829e-08, -7.28265e-06 | 1.32303e-16, "
    "9.44392e-17, -9.97002e-17, 1.00078e-16, -9.17977e-17, 1.17038e-16, "
    "-4.63278e-17, 7.43909e-17, 8.70943e-16, 7.20104e-14, -1.7372e-13, "
    "5.97612e-12, 4.58446e-12, 3.21306e-10, -5.15641e-09, -3.62788e-09, "
    "-2.49149e-07, 1.26824e-05, 0.000316752, 0.739324, -6.08281, -0.000868416, "
    "8.98534e-07, 1.18775e-08, -1.75726e-08, 5.13174e-11, -1.08752e-13, "
    "1.07653e-10, 4.88637e-11, 4.18285e-10, -4.85753e-08, 6.48908e-06 | "
    "-1.16355e-18, -1.7083e-18, -8.76569e-17, -1.07869e-16, 3.00518e-16, "
    "1.71216e-16, 2.52059e-16, 1.78676e-16, 8.12585e-17, 9.25124e-17, "
    "5.26408e-17, 1.01237e-16, -9.02945e-17, -2.92787e-16, -6.9502e-17, "
    "7.63305e-17, -5.50339e-17, 2.69944e-07, 3.38101e-06, 0.000317987, "
    "-0.000868416, 5.47115, -0.000985186, -3.3146e-06, 4.9039e-06, "
    "8.49636e-09, 2.15915e-11, 5.50747e-09, 8.92688e-10, -3.40083e-09, "
    "3.82378e-07, -1.93205e-05 | 9.68707e-17, 7.35802e-17, -2.60736e-16, "
    "8.15839e-17, -3.16652e-17, -2.02478e-16, 3.72915e-17, -1.78605e-16, "
    "2.38808e-16, 6.32037e-17, -1.90858e-16, 3.30113e-16, -1.69042e-18, "
    "4.96423e-17, -7.34703e-17, 4.02866e-17, 6.99168e-17, -2.79306e-10, "
    "-3.49827e-09, -3.29015e-07, 8.98534e-07, -0.000985186, -4.85082, "
    "-0.00050704, 0.000750159, -8.92179e-07, 4.30846e-10, -1.22123e-08, "
    "-3.08321e-09, -3.7757e-09, 3.15983e-07, -1.06259e-05 | -4.46105e-17, "
    "-5.53042e-17, 2.63342e-16, 9.33265e-17, -2.22121e-16, -1.21418e-16, "
    "-1.81e-16, 1.14509e-16, -8.04657e-17, 8.44899e-17, 7.03322e-17, "
    "-1.29385e-16, 6.88255e-17, -2.73597e-16, -9.36266e-16, 7.84628e-16, "
    "-2.8239e-14, -2.39692e-11, -1.33989e-10, -7.06788e-09, 1.18775e-08, "
    "-3.3146e-06, -0.00050704, -4.21482, 0.000997735, -0.000788018, "
    "9.49127e-08, -2.92178e-07, -2.08426e-08, -2.30748e-08, 1.12247e-06, "
    "-1.67313e-05 | -3.77646e-17, -5.86813e-17, 3.41237e-17, 1.67495e-16, "
    "-2.90037e-16, 2.22161e-16, -1.38115e-16, -4.15433e-17, -1.98115e-16, "
    "1.60703e-17, -2.95587e-16, -1.90612e-16, -2.30889e-16, 2.96035e-16, "
    "1.57744e-15, -1.37499e-15, 4.17107e-14, 3.54622e-11, 1.98235e-10, "
    "1.04568e-08, -1.75726e-08, 4.9039e-06, 0.000750159, 0.000997735, 4.17939, "
    "0.000502329, 7.09244e-07, 1.06562e-06, 5.46923e-08, -3.3476e-08, "
    "2.26441e-06, -2.19163e-05 | 3.4205e-17, 7.84958e-17, -2.27148e-17, "
    "-2.14049e-16, 1.7452e-16, -2.92042e-16, -6.27474e-17, 1.62212e-16, "
    "2.84323e-16, -3.90547e-17, 3.02822e-16, 3.29175e-16, 4.4922e-16, "
    "-9.47378e-17, -2.17896e-16, 7.27632e-17, -2.65807e-16, 1.64482e-13, "
    "-7.79814e-13, 2.45042e-11, 5.13098e-11, 8.49637e-09, -8.92179e-07, "
    "-0.000788018, 0.000502329, 3.56246, 0.000974257, 2.69589e-05, "
    "-4.46797e-07, -2.26517e-07, 7.49009e-06, -2.70308e-05 | 2.34499e-17, "
    "-5.58651e-17, 2.24727e-16, -4.70616e-17, -1.20869e-16, -1.68603e-16, "
    "-8.9011e-17, 2.98855e-16, 1.24949e-16, -2.79342e-16, 7.1293e-17, "
    "-1.10657e-16, -2.36548e-17, -2.98117e-16, -1.35988e-16, -1.92041e-16, "
    "-1.01064e-17, -3.22703e-15, 8.85713e-15, 5.55603e-14, -1.08215e-13, "
    "2.15959e-11, 4.30847e-10, 9.49126e-08, 7.09244e-07, 0.000974257, "
    "-2.88135, 0.000744412, -0.000290576, 5.3618e-06, -1.9514e-05, 3.35741e-05 "
    "| 3.47759e-17, -4.97264e-17, 1.5394e-16, -7.68138e-17, -8.69999e-19, "
    "9.79054e-17, -6.71777e-17, 3.95133e-16, 1.76716e-16, 1.73619e-15, "
    "-1.16917e-15, 6.24716e-15, 1.5168e-14, 7.06316e-14, -4.93668e-14, "
    "-1.17328e-13, -3.26289e-13, 1.81983e-11, -1.06826e-11, 2.09366e-10, "
    "1.07658e-10, 5.50747e-09, -1.22123e-08, -2.92178e-07, 1.06562e-06, "
    "2.69589e-05, 0.000744412, -2.17898, -0.000978873, -0.000552578, "
    "0.000179686, -6.57665e-05 | -1.19597e-17, 7.77192e-17, -4.10162e-17, "
    "-4.98866e-17, 2.77514e-16, -2.95375e-17, -3.99937e-17, 5.1325e-18, "
    "1.44018e-16, 1.07115e-15, -6.44665e-16, 4.60147e-15, 5.68546e-15, "
    "2.3582e-14, -3.3237e-14, -3.69994e-14, -2.10881e-13, 4.61884e-12, "
    "-5.90496e-12, 4.18405e-11, 4.88642e-11, 8.92686e-10, -3.08321e-09, "
    "-2.08426e-08, 5.46923e-08, -4.46797e-07, -0.000290576, -0.000978873, "
    "2.07259, 0.000624179, -0.000635245, 8.10784e-05 | 3.43422e-17, "
    "-7.6862e-17, 6.957e-14, 7.18314e-13, -1.32725e-12, -1.20181e-12, "
    "-1.11926e-12, -2.15527e-12, -8.85036e-13, 7.08482e-12, -3.22322e-12, "
    "9.29993e-12, -1.324e-11, -2.25972e-11, -1.353e-11, 1.74608e-11, "
    "-4.15104e-11, -2.39845e-10, -1.51658e-10, -5.92707e-10, 4.18281e-10, "
    "-3.40084e-09, -3.7757e-09, -2.30748e-08, -3.3476e-08, -2.26517e-07, "
    "5.3618e-06, -0.000552578, 0.000624179, 1.37088, -0.000675164, 0.000163467 "
    "| -4.05793e-16, -1.22011e-15, -8.67952e-12, -8.9668e-11, 1.65629e-10, "
    "1.50009e-10, 1.39673e-10, 2.69052e-10, 1.10459e-10, -8.83765e-10, "
    "4.01932e-10, -1.15901e-09, 1.6508e-09, 2.81415e-09, 1.68021e-09, "
    "-2.1712e-09, 5.13737e-09, 2.94329e-08, 1.82773e-08, 7.06829e-08, "
    "-4.85753e-08, 3.82378e-07, 3.15983e-07, 1.12247e-06, 2.26441e-06, "
    "7.49009e-06, -1.9514e-05, 0.000179686, -0.000635245, -0.000675164, "
    "-0.793108, 0.000425638 | 6.21092e-12, 1.36863e-11, 5.09877e-08, "
    "4.55672e-07, -9.00336e-07, -6.7344e-07, -6.67687e-07, -9.72766e-07, "
    "-3.46431e-07, 2.59296e-06, -9.72513e-07, 2.10394e-06, -2.50047e-06, "
    "-2.68317e-06, -1.4163e-06, 1.45467e-06, -3.04939e-06, -6.40908e-06, "
    "-3.96655e-06, -7.28265e-06, 6.48908e-06, -1.93205e-05, -1.06259e-05, "
    "-1.67313e-05, -2.19163e-05, -2.70308e-05, 3.35741e-05, -6.57665e-05, "
    "8.10784e-05, 0.000163467, 0.000425638, -0.0692168] (length=1024) [0 "
    "<repeats 1024 times>] (length=1024) CUDA";

#endif  // MATRIX3
