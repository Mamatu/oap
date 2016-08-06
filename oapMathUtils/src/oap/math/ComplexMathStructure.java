package ogla.math;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mmatula
 */
public class ComplexMathStructure extends MathStructure {

    private List<Complex> complexes = new ArrayList<Complex>();

    private ComplexMathStructure() {
    }

    public ComplexMathStructure(Complex complex) {
        this.complexes.add(complex);
    }

    public ComplexMathStructure(Complex[] complexes) {
        for (Complex complex : complexes) {
            this.complexes.add(complex);
        }
    }

    public ComplexMathStructure(List<Complex> complexes) {
        this.complexes.addAll(complexes);
    }

    public int getDimension() {
        return 1;
    }

    public int getSize(int index) {
        if (index != 0) {
            return -1;
        }
        return complexes.size();
    }

    public Complex getComplex(int[] index, Complex complex) {
        if (index.length == 1) {
            Complex ref = this.complexes.get(index[0]);
            complex.set(ref);
        }
        return complex;
    }

    public int getCount(int[] index) {
        if (index == null) {
            return this.complexes.size();
        }
        return -1;
    }

    @Override
    public MathStructure createCopy() {
        ComplexMathStructure cms = new ComplexMathStructure();
        for (Complex c : this.complexes) {
            cms.complexes.add((Complex) c.createCopy());
        }
        return cms;
    }

    public MathStructure copy(MathStructure mathStructure) {
        ComplexMathStructure cms = (ComplexMathStructure) mathStructure;
        for (int fa=0;fa<this.complexes.size();fa++) {
            this.complexes.get(fa).copy(cms.complexes.get(fa));
        }
        return cms;
    }
}
