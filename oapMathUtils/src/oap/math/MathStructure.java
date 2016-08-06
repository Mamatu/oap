/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.util.List;

/**
 *
 * @author mmatula
 */
public abstract class MathStructure {

    public interface Creator {

        public Brackets getBoundary();

        public boolean setParams(List<MathStructure> mathStructures) throws SyntaxErrorException;

        public MathStructure create();
    }

    public Complex getComplex(Complex out) {
        int[] index = {0};
        return this.getComplex(index, out);
    }

    private MathStructure.Creator creator = null;

    public MathStructure() {
    }

    public MathStructure(MathStructure.Creator creator) {
        this.creator = creator;
    }

    /**
     * Get dimensions of array in which are stored values.
     *
     * @return
     */
    abstract public int getDimension();

    /**
     * Get size of subdimension. Index - index of subdimension. For example: for
     * getDimension() = 3, getSize(0) - width, getSize(1) - height getSize(2) -
     * depth
     *
     * @param index
     * @return
     */
    abstract public int getSize(int index);

    abstract public Complex getComplex(int[] index, Complex out);

    abstract public MathStructure createCopy();

    abstract public MathStructure copy(MathStructure mathStructure);

    // abstract public MathStructure createCopy(MathStructure ms);
    public MathStructure.Creator getCreator() {
        return this.creator;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        final int dimensions = this.getDimension();
        int[] index = new int[dimensions];
        int[] sizes = new int[dimensions - 1];
        for (int fa = 0; fa < dimensions; fa++) {
            sizes[fa] = this.getSize(fa);
            index[fa] = 0;
        }
        int fa = 0;
        boolean stop = false;
        Complex complex = new Complex();
        if (this.getCreator().getBoundary() != null) {
            builder.append(this.getCreator().getBoundary().getLeftSymbol());
        }
        while (stop == false) {
            complex = this.getComplex(index, complex);
            builder.append(String.valueOf(complex));
            index[dimensions - 1]++;
            for (int fb = dimensions - 1; fb > 0; fb--) {
                if (index[dimensions - 1] == sizes[dimensions - 1]) {
                    index[dimensions - 1] = 0;
                    index[dimensions - 2]++;
                    if (dimensions - 2 == -1 && index[dimensions - 1] == 0) {
                        if (this.getCreator().getBoundary() != null) {
                            builder.append(this.getCreator().getBoundary().getRightSymbol());
                        }
                        break;
                    }
                }
            }
        }
        return builder.toString();
    }
}
