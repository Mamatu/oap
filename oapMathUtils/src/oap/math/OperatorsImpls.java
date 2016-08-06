/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

/**
 *
 * @author mmatula
 */
class OperatorsImpls {

    static class Seprator implements Operator {

        public Seprator() {
        }

        public int getWeight() {
            return 0;
        }

        public char getSymbol() {
            return ',';
        }

        private Complex param1 = null;
        private Complex param2 = null;

        public boolean setParams(MathStructure param1, MathStructure param2) {
            if (param1 instanceof Complex && param2 instanceof Complex) {
                this.param1 = new Complex((Complex) param1);
                this.param2 = new Complex((Complex) param2);
                return true;
            }
            return false;
        }

        private Complex[] complexes = new Complex[2];

        /**
         *
         * @param param1 first parameter and output of calculations
         * @param param2 second parameter
         * @return
         * @throws SyntaxErrorException
         */
        public MathStructure execute() throws SyntaxErrorException {
            complexes[0] = param1;
            complexes[1] = param2;
            MathStructure mathStructure = new ComplexMathStructure(complexes);
            return mathStructure;
        }

        public String toString() {
            return String.format("Operator: %c", this.getSymbol());
        }

    }

    static class Addition implements Operator {

        public Addition() {
        }

        public int getWeight() {
            return 1;
        }

        public char getSymbol() {
            return '+';
        }

        private MathStructure param1 = null;
        private MathStructure param2 = null;

        public boolean setParams(MathStructure param1, MathStructure param2) {
            boolean b = (param1 instanceof Matrix && param2 instanceof Matrix)
                    || (param1 instanceof Complex && param2 instanceof Complex);
            if (b) {
                this.param1 = param1;
                this.param2 = param2;
            } else {
                this.param1 = null;
                this.param2 = null;

            }
            return b;
        }

        public MathStructure execute() throws SyntaxErrorException {
            if (param1 == null && param2 == null) {
                throw new SyntaxErrorException("");
            }
            if (param1 instanceof Matrix && param2 instanceof Matrix) {
                Matrix matrix1 = (Matrix) param1;
                Matrix matrix2 = (Matrix) param2;
                if (matrix1.rows() != matrix2.rows()) {
                    throw new SyntaxErrorException("matrices : bad dimesnions (rows)");
                }
                if (matrix1.columns() != matrix2.columns()) {
                    throw new SyntaxErrorException("matrices : bad dimesnions (columns)");
                }
                matrix1.add(matrix2);
                return matrix1;
            } else if (param1 instanceof Complex && param2 instanceof Complex) {
                Complex complex1 = (Complex) param1;
                Complex complex2 = (Complex) param2;
                complex1.add(complex2);
                return complex1;
            }
            return null;
        }

        public String toString() {
            return String.format("Operator: %c", this.getSymbol());
        }
    }

    static class Substraction implements Operator {

        public Substraction() {
        }

        public int getWeight() {
            return 1;
        }
        private MathStructure param1 = null;
        private MathStructure param2 = null;

        public char getSymbol() {
            return '-';
        }

        public boolean setParams(MathStructure param1, MathStructure param2) {
            boolean b = (param1 instanceof Matrix && param2 instanceof Matrix)
                    || (param1 instanceof Complex && param2 instanceof Complex)
                    || (param2 instanceof Complex);
            if (b) {
                this.param1 = param1;
                this.param2 = param2;
            } else {
                this.param1 = null;
                this.param2 = null;

            }
            return b;
        }

        public MathStructure execute() throws SyntaxErrorException {
            if (param1 == null && param2 == null) {
                throw new SyntaxErrorException("");
            }
            if (param1 instanceof Matrix && param2 instanceof Matrix) {
                Matrix matrix1 = (Matrix) param1;
                Matrix matrix2 = (Matrix) param2;
                if (matrix1.rows() != matrix2.rows()) {
                    throw new SyntaxErrorException("matrices : bad dimesnions (rows)");
                }
                if (matrix1.columns() != matrix2.columns()) {
                    throw new SyntaxErrorException("matrices : bad dimesnions (columns)");
                }
                matrix1.substract(matrix2);
                return matrix1;
            } else if (param1 instanceof Complex && param2 instanceof Complex) {
                Complex complex1 = (Complex) param1;
                Complex complex2 = (Complex) param2;
                complex1.substract(complex2);
                return complex1;
            } else if (param1 == null && param2 instanceof Complex) {
                Complex complex2 = (Complex) param2;
                complex2.multiply(-1);
                return complex2;
            }
            return null;
        }

        public String toString() {
            return String.format("Operator: %c", this.getSymbol());
        }
    }

    static class Multiplication implements Operator {

        public Multiplication() {
        }

        public int getWeight() {
            return 2;
        }

        public char getSymbol() {
            return '*';
        }
        private MathStructure param1 = null;
        private MathStructure param2 = null;

        public boolean setParams(MathStructure param1, MathStructure param2) {
            boolean b = (param1 instanceof Matrix && param2 instanceof Matrix)
                    || (param1 instanceof Complex && param2 instanceof Matrix)
                    || (param1 instanceof Matrix && param2 instanceof Complex)
                    || (param1 instanceof Complex && param2 instanceof Complex);
            if (b) {
                this.param1 = param1;
                this.param2 = param2;
            } else {
                this.param1 = null;
                this.param2 = null;

            }
            return b;
        }

        public MathStructure execute() throws SyntaxErrorException {
            if (param1 == null && param2 == null) {
                throw new SyntaxErrorException("");
            }
            if (param1 instanceof Matrix && param2 instanceof Matrix) {
                Matrix matrix1 = (Matrix) param1;
                Matrix matrix2 = (Matrix) param2;
                if (matrix1.rows() != matrix2.columns()) {
                    throw new SyntaxErrorException("Invalid");
                }
                matrix1.multiply(matrix2);
                return matrix1;
            } else if (param1 instanceof Complex && param2 instanceof Complex) {
                Complex complex1 = (Complex) param1;
                Complex complex2 = (Complex) param2;
                complex1.multiply(complex2);
                return complex1;
            } else if ((param1 instanceof Complex && param2 instanceof Matrix)) {
                Complex complex1 = (Complex) param1;
                Matrix matrix2 = (Matrix) param2;
                matrix2.multiply(complex1);
                return matrix2;
            } else if ((param1 instanceof Matrix && param2 instanceof Complex)) {
                Complex complex2 = (Complex) param2;
                Matrix matrix1 = (Matrix) param1;
                matrix1.multiply(complex2);
                return matrix1;
            }
            return null;
        }

        public String toString() {
            return String.format("Operator: %c", this.getSymbol());
        }
    }

    static class Division implements Operator {

        public Division() {
        }

        public int getWeight() {
            return 2;
        }

        public char getSymbol() {
            return '/';
        }
        private MathStructure param1 = null;
        private MathStructure param2 = null;

        public boolean setParams(MathStructure param1, MathStructure param2) {
            boolean b = (param1 instanceof Complex && param2 instanceof Complex);
            if (b) {
                this.param1 = param1;
                this.param2 = param2;
            } else {
                this.param1 = null;
                this.param2 = null;

            }
            return b;
        }

        public MathStructure execute() throws SyntaxErrorException {
            if (param1 == null && param2 == null) {
                throw new SyntaxErrorException("");
            }
            if (param1 instanceof Complex && param2 instanceof Complex) {
                Complex complex1 = (Complex) param1;
                Complex complex2 = (Complex) param2;
                complex1.divide(complex2);
                return complex1;
            }
            return null;
        }

        public String toString() {
            return String.format("Operator: %c", this.getSymbol());
        }
    }

    static class Power implements Operator {

        public Power() {
        }

        public int getWeight() {
            return 3;
        }

        public char getSymbol() {
            return '^';
        }
        private MathStructure param1 = null;
        private MathStructure param2 = null;

        public boolean setParams(MathStructure param1, MathStructure param2) {
            boolean b = (param1 instanceof Complex && param2 instanceof Complex);
            if (b) {
                this.param1 = param1;
                this.param2 = param2;
            } else {
                this.param1 = null;
                this.param2 = null;

            }
            return b;
        }

        public MathStructure execute() throws SyntaxErrorException {
            if (param1 == null && param2 == null) {
                throw new SyntaxErrorException("");
            }
            if (param1 instanceof Complex && param2 instanceof Complex) {
                Complex complex1 = (Complex) param1;
                Complex complex2 = (Complex) param2;
                float po = (float) Math.pow(complex1.re.doubleValue(), complex2.re.doubleValue());
                complex1.set(po);
                return complex1;
            }
            return null;
        }

        public String toString() {
            return String.format("Operator: %c", this.getSymbol());
        }
    }
}
