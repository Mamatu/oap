/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author mmatula
 */
public class ParserImpl implements Parser {

    private List<Operator> operators = new ArrayList<Operator>();
    private List<MathStructure.Creator> mathStructuresCreators = new ArrayList<MathStructure.Creator>();
    private List<Function> functions = new ArrayList<Function>();
    private List<Brackets> brackets = new ArrayList<Brackets>();
    private Map<String, MathStructure> variables = new HashMap<String, MathStructure>();

    public void setVariableValue(String name, MathStructure complex) {
        this.variables.put(name, complex);
    }

    public void setVariable(String name) {
        this.variables.put(name, null);
    }

    public void removeVariable(String name) {
        this.variables.remove(name);
    }

    public void addOperator(Operator operator) {
        this.operators.add(operator);
    }

    public void removeOperator(Operator operator) {
        this.operators.remove(operator);
    }

    public void addBracket(Brackets bracket) {
        if (bracket.getLeftSymbol() == bracket.getRightSymbol()) {
        }
        this.brackets.add(bracket);
    }

    public void removeBracket(Brackets bracket) {
        this.brackets.remove(bracket);
    }

    public void addMathStructureCreator(MathStructure.Creator creator) {
        this.mathStructuresCreators.add(creator);
    }

    public void removeMathStructureCreator(MathStructure.Creator creator) {
        this.mathStructuresCreators.remove(creator);
    }

    public void addFunction(Function function) {
        this.functions.add(function);
    }

    public void removeFunction(Function function) {
        this.functions.remove(function);
    }

    private int findClosingBracket(Brackets bracket, String equation, int begin) throws SyntaxErrorException {
        int index = -1;
        int stackCounter = 0;
        for (int fa = begin; fa < equation.length(); fa++) {
            if (equation.charAt(fa) == bracket.getLeftSymbol()) {
                stackCounter++;
            }
            if (equation.charAt(fa) == bracket.getRightSymbol()) {
                stackCounter--;
                if (stackCounter == 0) {
                    index = fa;
                    break;
                }
            }
        }
        if (index == -1) {
            throw new SyntaxErrorException("");
        }
        return index;
    }

    private void insert(List<Object> objs, int index, Object obj) {
        Object e = obj;
        for (int fa = index; fa < objs.size(); fa++) {
            e = objs.set(fa, e);
        }
        objs.add(e);
    }
    private Map<Object, String> codes = new IdentityHashMap<Object, String>();

    private void parse(Code code) throws SyntaxErrorException {
        int na = 0;
        do {
            Code.Item obj = code.get(na);
            if (obj.getType() == Code.Type.EQUATION_ITEM) {
                EquationItem equationItem = (EquationItem) obj;
                String equation = equationItem.string;
                equation = equation.replaceAll("\\s+", "");
                int fa = 0;
                boolean parsed = true;
                do {
                    parsed = true;
                    if (this.parseComplex(equation, fa, code, na) == false) {
                        if (this.parseVariable(equation, fa, code, na) == false) {
                            if (this.parseBrackets(equation, fa, code, na) == false) {
                                if (this.parseMathStructure(equation, fa, code, na) == false) {
                                    if (this.parseOperators(equation, fa, code, na) == false) {
                                        if (this.parseFunctions(equation, fa, code, na) == false) {
                                            parsed = false;
                                            fa++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } while (parsed == false);
                if (fa > 0) {
                    code.insert(na, new StringItem(equation.substring(0, fa)));
                }
                na = 0;
            } else {
                na++;
            }
        } while (na != code.size());
    }

    private int findOperatorsList(Code code, int index) {
        for (int fa = index; fa < code.size(); fa++) {
            if (code.get(fa).getType() == Code.Type.OPERATOR_ITEMS) {
                return fa;
            }
        }
        return -1;
    }

    private List<Integer> operatorsIntegersList = new ArrayList<Integer>();

    private Integer[] examineOperators(OperatorItem.List operatorList, MathStructure param1, MathStructure param2) {
        operatorsIntegersList.clear();
        for (int fa = 0; fa < operatorList.size(); fa++) {
            Operator operator = operatorList.get(fa).operator;
            if (operator.setParams(param1, param2) == true) {
                operatorsIntegersList.add(fa);
            }
        }
        if (operatorsIntegersList.isEmpty()) {
            return null;
        }
        Integer[] array = new Integer[operatorsIntegersList.size()];
        return operatorsIntegersList.toArray(array);
    }

    private Operator getMostImportantOperator(OperatorItem.List operatorList, Integer[] operatorIndieces) {
        Operator operator = null;
        int weight = -1;
        for (Integer fa : operatorIndieces) {
            if (operatorList.get(fa).operator.getWeight() > weight) {
                weight = operatorList.get(fa).operator.getWeight();
                operator = operatorList.get(fa).operator;
            }
        }
        return operator;
    }

    private MathStructure getParam(Code code, int index) throws SyntaxErrorException {
        if (index < 0 || index > code.size()) {
            return null;
        }
        MathStructure param1 = null;
        final boolean pb1 = (index >= 0);
        if (pb1) {
            Code.Item objParam1 = code.get(index);
            if (objParam1.getType() == Code.Type.BRACKET_ITEM) {
                this.executeBracket(code, index);
            } else if (objParam1.getType() == Code.Type.FUNCTION_ITEMS) {
                this.executeFunctionsList(code, index);
            }
            objParam1 = code.get(index);
            if (objParam1.getType() == Code.Type.MATH_STRUCTURE_ITEM) {
                param1 = ((MathStructureItem) objParam1).mathStructure;
            } else if (objParam1.getType() == Code.Type.VARIABLE_ITEM) {
                Variable v = ((VariableItem) objParam1).variable;
                Variable.updateVariable(v, variables);
                param1 = v.mathStructure;
            }
        }
        return param1;
    }

    private void executeOperatorsList(Code code, int index) throws SyntaxErrorException {
        try {
            Object object = code.get(index);
            OperatorItem.List operatorsList = (OperatorItem.List) object;
            int nindex = this.findOperatorsList(code, index + 1);
            if (nindex > -1) {
                OperatorItem.List nextOperatorsList = (OperatorItem.List) code.get(nindex);
                if (operatorsList.get(0).operator.getWeight() < nextOperatorsList.get(0).operator.getWeight()) {
                    this.executeOperatorsList(code, nindex);
                }
            }
            MathStructure param1 = getParam(code, index - 1);
            MathStructure param2 = getParam(code, index + 1);
            MathStructure cparam1 = null;
            if (param1 != null) {
                cparam1 = param1.createCopy();
            }
            MathStructure cparam2 = null;
            if (param2 != null) {
                cparam2 = param2.createCopy();
            }
            Integer[] operatorsIndieces = this.examineOperators(operatorsList, cparam1, cparam2);
            Operator operator = this.getMostImportantOperator(operatorsList, operatorsIndieces);
            if (operator.setParams(cparam1, cparam2)) {
                MathStructure result = operator.execute();
                if (cparam1 != null) {
                    code.set(index - 1, new MathStructureItem(result));
                    code.remove(index);
                    if (cparam2 != null) {
                        code.remove(index);
                    }
                } else {
                    code.set(index, new MathStructureItem(result));
                    if (cparam2 != null) {
                        code.remove(index + 1);
                    }
                }
            }
        } catch (SyntaxErrorException see) {
            see.setInvalidCode(codes.get(code.get(index)) + see.getInvalidCode());
            throw see;
        }
    }

    private void executeFunctionsList(Code code, int index) throws SyntaxErrorException {
        try {
            if (code.size() - 1 == index) {
            }
            FunctionItem.List functionsList = (FunctionItem.List) code.get(index);
            if (code.get(index + 1).getType() == Code.Type.BRACKET_ITEM) {
                Code codeInBrackets = ((BracketsItem) code.get(index + 1)).getCode();
                List<MathStructure> params = new ArrayList<MathStructure>();
                boolean mustBeRawCodeFunction = false;
                try {
                    MathStructure[] ms = this.execute1(codeInBrackets);
                    params.addAll(Arrays.asList(ms));
                } catch (Exception ex) {
                    mustBeRawCodeFunction = true;
                }
                for (int fa = 0; fa < functionsList.size(); fa++) {
                    boolean canBeContinued = false;
                    boolean hasRawCodeParams = functionsList.get(fa).function.rawCodeAsParams();
                    if ((mustBeRawCodeFunction == true && hasRawCodeParams == true) || mustBeRawCodeFunction == false) {
                        canBeContinued = true;
                    }
                    if (hasRawCodeParams == false) {
                        if (functionsList.get(fa).function.getParamsCount() == params.size()) {
                            if (functionsList.get(fa).function.setParams(params, null) == true) {
                                MathStructure object = functionsList.get(fa).function.execute();
                                code.set(index, new MathStructureItem(object));
                                code.remove(index + 1);
                                break;
                            }
                        }
                    } else {
                        List<ParamItem> paramsItems = new ArrayList<ParamItem>();
                        Code tempCode = new Code();
                        for (int fb = 0; fb < codeInBrackets.size(); fb++) {
                            if (codeInBrackets.get(fb).getType() == Code.Type.OPERATOR_ITEMS) {
                                OperatorItem.List operatorItems = (OperatorItem.List) codeInBrackets.get(fb);
                                if (operatorItems.size() == 1) {
                                    if (operatorItems.get(0).operator instanceof OperatorsImpls.Seprator) {
                                        paramsItems.add(new ParamItem(tempCode));
                                        tempCode.clear();
                                    } else {
                                        tempCode.add(codeInBrackets.get(fb));
                                    }
                                } else {
                                    tempCode.add(codeInBrackets.get(fb));
                                }
                            } else {
                                tempCode.add(codeInBrackets.get(fb));
                            }
                        }
                        paramsItems.add(new ParamItem(tempCode));
                        tempCode.clear();
                        if (functionsList.get(fa).function.setParams(paramsItems, this) == true) {
                            MathStructure object = functionsList.get(fa).function.execute();
                            code.set(index, new MathStructureItem(object));
                            code.remove(index + 1);
                            break;
                        } else {
                        }
                    }
                }
            }
        } catch (SyntaxErrorException see) {
            see.setInvalidCode(codes.get(code.get(index)) + see.getInvalidCode());
            throw see;
        }
    }

    private void executeBracket(Code code, int index) throws SyntaxErrorException {
        try {
            BracketsItem bracketItem = (BracketsItem) code.get(index);
            Code subCode = bracketItem.getCode();
            MathStructure[] mathStructures = this.execute1(subCode);
            code.set(index, new MathStructureItem(mathStructures[0]));
            for (int fa = 1; fa < mathStructures.length; fa++) {
                code.insert(index + fa, new MathStructureItem(mathStructures[fa]));
            }
        } catch (SyntaxErrorException see) {
            see.setInvalidCode(codes.get(code.get(index)));
            throw see;
        }
    }

    private void executeMathStructure(Code code, int index) throws SyntaxErrorException {
        try {
            List<MathStructure> mathStructures = new ArrayList<MathStructure>();
            mathStructures.clear();
            MathStructureCreatorItem.List creatorItems = (MathStructureCreatorItem.List) code.get(index);
            MathStructure.Creator creator = creatorItems.get(0).creator;
            Code objs = creatorItems.code;
            MathStructure[] mathStructuresArray = this.execute1(objs);
            MathStructure mathStructure1 = null;
            mathStructures.addAll(Arrays.asList(mathStructuresArray));
            if (creator.setParams(mathStructures) == true) {
                mathStructure1 = creator.create();
            }
            code.remove(index);
            code.set(index, new MathStructureItem(mathStructure1));
        } catch (SyntaxErrorException see) {
            final Object object = code.get(index + 1);
            final String scode = this.codes.get(object);
            see.setInvalidCode(scode);
            throw see;
        }
    }

    private boolean parseComplex(String equation, int fa, Code code, int na) {
        Complex complex = null;
        Complex complex1 = null;
        int complexEndIndex = 0;
        for (int fb = fa + 1; fb <= equation.length() + 1 && (complex1 != null || fb == fa + 1); fb++) {
            complex = complex1;
            if (fb <= equation.length()) {
                String substring = equation.substring(fa, fb);
                complex1 = Complex.parseComplex(substring);
            }
            complexEndIndex = fb;
        }
        if (complex != null) {
            code.set(na, new MathStructureItem(complex));
            codes.put(complex, complex.toString());
            if (complexEndIndex < equation.length()) {
                complexEndIndex--;
                String part2 = equation.substring(complexEndIndex, equation.length());
                code.insert(na + 1, new EquationItem(part2));
                equation = part2;
            }
        }
        return (complex != null);
    }

    private boolean parseVariable(String equation, int fa, Code code, int na) {
        String variableName = null;
        String variableName1 = null;
        int variableEndIndex = 0;
        for (int fb = fa + 1; fb <= equation.length() + 1 && (variableName1 != null || fb == fa + 1); fb++) {
            variableName = variableName1;
            if (fb <= equation.length()) {
                String substring = equation.substring(fa, fb);
                if (this.variables.containsKey(substring)) {
                    variableName1 = substring;
                } else {
                    variableName1 = null;
                }
            }
            variableEndIndex = fb;
        }
        if (variableName != null) {
            Variable variable = new Variable(variableName);
            code.set(na, new VariableItem(variable));
            codes.put(variable, variable.toString());
            if (variableEndIndex < equation.length()) {
                variableEndIndex--;
                String part2 = equation.substring(variableEndIndex, equation.length());
                code.insert(na + 1, new EquationItem(part2));
                equation = part2;
            }
        }
        return (variableName != null);
    }

    private boolean parseBrackets(String equation, int fa, Code code, int na) throws SyntaxErrorException {
        for (Brackets bracket : this.brackets) {
            char symbol = equation.charAt(fa);
            if (symbol == bracket.getLeftSymbol()) {
                int index1 = findClosingBracket(bracket, equation, fa);
                String part2 = equation.substring(fa + 1, index1);
                String part3 = equation.substring(index1 + 1, equation.length());
                //objs.set(na, bracket);
                Code subCode = new Code();
                subCode.add(new EquationItem(part2));
                this.parse(subCode);
                codes.put(subCode, part2);
                code.set(na, new BracketsItem(bracket, subCode));
                if (part3.length() > 0) {
                    code.insert(na + 1, new EquationItem(part3));
                }
                fa = 0;
                return true;
            }
        }
        return false;
    }

    private boolean parseMathStructure(String equation, int fa, Code code, int na) throws SyntaxErrorException {
        boolean status = false;
        boolean isFirst = true;
        MathStructureCreatorItem.List creatorList = null;
        for (MathStructure.Creator mathStructureCreator : this.mathStructuresCreators) {
            int fa1 = fa;
            char symbol = equation.charAt(fa1);
            if (isFirst == true) {
                if (symbol == mathStructureCreator.getBoundary().getLeftSymbol()) {
                    int index1 = this.findClosingBracket(mathStructureCreator.getBoundary(), equation, fa1);
                    String part2 = equation.substring(fa1 + 1, index1);
                    String part3 = equation.substring(index1 + 1, equation.length());
                    Code subCode = new Code();
                    subCode.add(new EquationItem(part2));
                    this.parse(subCode);
                    MathStructureCreatorItem.List creatorList1 = new MathStructureCreatorItem.List(subCode);
                    creatorList1.add(new MathStructureCreatorItem(mathStructureCreator));
                    creatorList = creatorList1;
                    code.set(na, creatorList1);
                    if (part3.length() > 0) {
                        code.insert(na + 1, new EquationItem(part3));
                    }
                    isFirst = false;
                    status = true;
                }
            } else {
                if (creatorList.get(0).creator.getBoundary().getLeftSymbol() == mathStructureCreator.getBoundary().getLeftSymbol()
                        && creatorList.get(0).creator.getBoundary().getRightSymbol() == mathStructureCreator.getBoundary().getRightSymbol()) {
                    creatorList.add(new MathStructureCreatorItem(mathStructureCreator));
                }
            }
        }
        return status;
    }

    private boolean parseOperators(String equation, int fa, Code code, int na) {
        OperatorItem.List operatorsList = null;
        for (Operator operator : operators) {
            char symbol = equation.charAt(fa);
            if (symbol == operator.getSymbol()) {
                if (operatorsList == null) {
                    operatorsList = new OperatorItem.List();
                }
                operatorsList.add(new OperatorItem(operator));
            }
        }
        if (operatorsList != null) {
            String sign = equation.substring(fa, fa + 1);
            String part2 = equation.substring(fa + 1, equation.length());
            code.set(na, operatorsList);
            code.insert(na + 1, new EquationItem(part2));
            equation = part2;
            codes.put(operatorsList, sign);
        }
        return (operatorsList != null);
    }

    private boolean parseFunctions(String equation, int fa, Code code, int na) {
        FunctionItem.List functionsList = null;
        for (Function function : functions) {
            if (fa + function.getName().length() < equation.length()) {
                String comp = equation.substring(fa, fa + function.getName().length());
                if (comp.equals(function.getName())) {
                    if (functionsList == null) {
                        functionsList = new FunctionItem.List();
                    }
                    functionsList.add(new FunctionItem(function));
                }
            }
        }
        if (functionsList != null) {
            String functionName = equation.substring(fa, fa + functionsList.get(0).function.getName().length());
            String part2 = equation.substring(fa + functionsList.get(0).function.getName().length(), equation.length());
            code.set(na, functionsList);
            code.insert(na + 1, new EquationItem(part2));
            codes.put(functionsList, functionName);
        }
        return (functionsList != null);
    }

    public MathStructure[] execute(Object codeObject) throws SyntaxErrorException {
        Code code1 = (Code) codeObject;
        Code code = code1.createCopy();
        MathStructure[] ms = execute1(code);
        return ms;
    }

    private MathStructure[] execute1(Code code) throws SyntaxErrorException {
        try {
            boolean next = true;
            while (next) {
                next = false;
                int fa = 0;
                while (fa < code.size()) {
                    Code.Item object = code.get(fa);
                    if (object.getType() == Code.Type.STRING_ITEM) {
                        StringItem stringItem = (StringItem) object;
                        if (variables.containsKey(stringItem.string)) {
                            MathStructure mathStructure = variables.get(stringItem.string);
                            if (mathStructure != null) {
                                code.set(fa, new VariableItem(new Variable(stringItem.string)));
                            }
                        }
                    } else if (object.getType() == Code.Type.OPERATOR_ITEMS) {
                        next = true;
                        this.executeOperatorsList(code, fa);
                        fa = 0;
                    } else if (object.getType() == Code.Type.FUNCTION_ITEMS) {
                        next = true;
                        this.executeFunctionsList(code, fa);
                        fa = 0;
                    } else if (object.getType() == Code.Type.BRACKET_ITEM) {
                        next = true;
                        this.executeBracket(code, fa);
                        fa = 0;
                    } else if (object.getType() == Code.Type.MATH_STRUCTURE_CREATOR_ITEMS) {
                        next = true;
                        this.executeMathStructure(code, fa);
                        fa = 0;
                    }
                    fa++;
                }
            }
            List<MathStructure> mathStructures = new ArrayList<MathStructure>();
            for (int fa = 0; fa < code.size(); fa++) {
                if (code.get(fa).getType() == Code.Type.VARIABLE_ITEM) {
                    VariableItem variableItem = (VariableItem) code.get(fa);
                    Variable.updateVariable(variableItem.variable, variables);
                    mathStructures.add(variableItem.variable.mathStructure);
                }
                if (code.get(fa).getType() == Code.Type.MATH_STRUCTURE_ITEM) {
                    MathStructure mathStructure = ((MathStructureItem) code.get(fa)).mathStructure;
                    mathStructures.add(mathStructure);
                }
            }
            MathStructure[] a = new MathStructure[mathStructures.size()];
            return mathStructures.toArray(a);
        } catch (SyntaxErrorException see) {
            throw see;
        }
    }

    public Object parse(String equation) throws SyntaxErrorException {
        Code code = new Code();
        code.add(new EquationItem(equation));
        this.parse(code);
        return code;
    }

    public MathStructure[] calculate(String equation) throws SyntaxErrorException {
        MathStructure[] out = null;
        codes.clear();
        Code code = new Code();
        code.add(new EquationItem(equation));
        this.parse(code);
        out = this.execute1(code);
        return out;
    }
}
