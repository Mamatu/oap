package ogla.math.test;

import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.math.MathStructure;
import ogla.math.ParserCreator;
import ogla.math.ParserImpl;

public class ComplexMathTest {

    public static void main(String[] args) {
        try {
            String e = "sum(exp(n+1),n,0,5-1)";
            ParserImpl parser = ParserCreator.create();
            Object code = parser.parse(e);
            MathStructure[] ms = parser.execute(code);
            System.out.println(ms[0].toString());
        } catch (ogla.math.SyntaxErrorException ex) {
            Logger.getLogger(ComplexMathTest.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}
