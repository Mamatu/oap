/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author mmatula
 */
public class ParserCreator {

    private static Map<Function, List<MathStructure>> mapFunctions = new HashMap<Function, List<MathStructure>>();

    private static void AddFunctions(ParserImpl parser) {
        Class clazz = ComplexMath.class;

        for (final Method method : clazz.getDeclaredMethods()) {

            parser.addFunction(new Function() {

                public int getParamsCount() {
                    return method.getGenericParameterTypes().length;
                }

                public String getName() {
                    return method.getName();
                }

                public boolean setParams(Object params, Parser parser) {
                    List<MathStructure> params1 = (List<MathStructure>) params;
                    if (params1.size() == 1 && params1.get(0).getDimension() == 1 && params1.get(0).getSize(0) == 1) {
                        mapFunctions.put(this, params1);
                        return true;
                    }
                    return false;
                }

                private Complex complex = new Complex();

                public MathStructure execute() throws SyntaxErrorException {
                    MathStructure list = mapFunctions.get(this).get(0);
                    Object result = null;
                    try {
                        int[] index = {0};
                        this.complex = list.getComplex(index, this.complex);
                        result = method.invoke(null, complex);
                    } catch (IllegalAccessException iae) {
                    } catch (IllegalArgumentException iae) {
                    } catch (InvocationTargetException iie) {
                    }

                    return (MathStructure) result;
                }

                @Override
                public String toString() {
                    return String.format("Function: %s", this.getName());
                }

                public boolean rawCodeAsParams() {
                    return false;
                }

                public String getErrorString() {
                    return "Bad dimension or size.";
                }
            }
            );
        }

        parser.addFunction(new Function() {

            @Override
            public String toString() {
                return this.getName();
            }

            public int getParamsCount() {
                return 4;
            }

            public String getName() {
                return "sum";
            }

            public boolean rawCodeAsParams() {
                return true;
            }

            private String error = "";
            private Parser parser = null;

            public boolean setParams(Object params, Parser parser) {
                class Extracter {

                    public String error = "";
                    public Parser parser = null;
                    public Complex output = null;

                    Extracter(Parser parser) {
                        this.parser = parser;
                    }

                    boolean extract(Code code) {
                        if (code.size() > 1) {
                            try {
                                MathStructure[] ms = parser.execute(code);
                                if (ms.length == 1 && (ms[0] instanceof Complex) == true) {
                                    this.output = (Complex) ms[0];
                                } else {
                                    return false;
                                }
                            } catch (SyntaxErrorException ex) {
                                Logger.getLogger(ParserCreator.class.getName()).log(Level.SEVERE, null, ex);
                            }
                        } else {
                            Code.Item itemComplexItem1 = code.get(0);
                            if (itemComplexItem1.getType() != Code.Type.MATH_STRUCTURE_ITEM) {
                                error = "Params are not integers.";
                                return false;
                            }
                            MathStructureItem msitem1 = (MathStructureItem) itemComplexItem1;
                            if ((msitem1.mathStructure instanceof Complex) == false) {
                                error = "Params are not integers.";
                                return false;
                            }
                            this.output = (Complex) msitem1.mathStructure;
                        }
                        return true;
                    }
                }
                error = "";
                List<ParamItem> paramItems = (List<ParamItem>) params;
                if (paramItems.size() != 4) {
                    error = "Invalid number of params (should be 4).";
                    return false;
                }
                Code.Item itemStringItem = paramItems.get(1).getCode().get(0);
                if (itemStringItem.getType() != Code.Type.STRING_ITEM) {
                    error = "The second param must be string.";
                    return false;
                }
                StringItem stringItem = (StringItem) itemStringItem;

                Extracter extracter = new Extracter(parser);
                if (extracter.extract(paramItems.get(2).getCode()) == false) {
                    return false;
                }
                Complex complex1 = extracter.output;
                if (extracter.extract(paramItems.get(3).getCode()) == false) {
                    return false;
                }
                Complex complex2 = extracter.output;
                if (complex1.im.floatValue() != 0 || complex2.im.floatValue() != 0) {
                    error = "Params are not integers.";
                    return false;
                }
                if (complex1.re.floatValue() != Math.round(complex1.re.floatValue()) || complex2.re.floatValue() != Math.round(complex2.re.floatValue())) {
                    error = "Params are not integers.";
                    return false;
                }
                int min = complex1.re.intValue();
                int max = complex2.re.intValue();
                if (min > max) {
                    error = "Min is greater than max (the third argument is min, the last is max)";
                    return false;
                }
                this.paramItem = paramItems.get(0);
                this.min = min;
                this.max = max;
                this.parser = parser;
                this.variableName = stringItem.string;
                return true;
            }

            private ParamItem paramItem = null;
            private String variableName = "";
            private int min = 0;
            private int max = 0;
            private OperatorsImpls.Addition addition = new OperatorsImpls.Addition();

            public MathStructure execute() throws SyntaxErrorException {
                Complex complex = new Complex();
                MathStructure param1 = null;
                MathStructure param2 = null;
                for (int fa = min; fa < max; fa++) {
                    Code code = this.paramItem.getCode().createCopy();
                    complex.set(fa);
                    this.parser.setVariableValue(this.variableName, complex);
                    MathStructure[] mathStructures = this.parser.execute(code);
                    if (fa == min && mathStructures.length == 1) {
                        param1 = mathStructures[0];
                    }
                    if (fa > min && mathStructures.length == 1) {
                        param2 = mathStructures[0];
                    }
                    if (param1 != null && param2 != null) {
                        addition.setParams(param1, param2);
                        param1 = addition.execute();
                    }
                }
                return param1;
            }

            public String getErrorString() {
                return error;
            }
        });
    }

    private static void AddOperators(ParserImpl parser) {
        Class clazz = OperatorsImpls.class;
        for (final Class clazz1 : clazz.getDeclaredClasses()) {
            try {
                java.lang.reflect.Constructor constructor = clazz1.getConstructor();
                Operator operator = (Operator) constructor.newInstance();
                parser.addOperator(operator);
            } catch (NoSuchMethodException ex) {
                System.out.println(ex.getMessage());
            } catch (InstantiationException ex) {
                System.out.println(ex.getMessage());
            } catch (IllegalAccessException ex) {
                System.out.println(ex.getMessage());
            } catch (InvocationTargetException ex) {
                System.out.println(ex.getMessage());
            }
        }
    }

    private static void AddMathStructures(ParserImpl parser) {
        parser.addMathStructureCreator(new Matrix.Craetor());
    }

    private static void AddBrackets(ParserImpl parser) {
        parser.addBracket(new Brackets() {

            public char getLeftSymbol() {
                return '(';
            }

            public char getRightSymbol() {
                return ')';
            }
        });
    }

    public static ParserImpl create() {
        ParserImpl parser = new ParserImpl();
        ParserCreator.AddFunctions(parser);
        ParserCreator.AddOperators(parser);
        ParserCreator.AddMathStructures(parser);
        ParserCreator.AddBrackets(parser);
        return parser;
    }
}
