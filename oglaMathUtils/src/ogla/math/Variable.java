package ogla.math;

import java.util.Map;

public class Variable {

    public String name = "";
    MathStructure mathStructure = null;

    public Variable(String name) {
        this.name = name;
    }

    public String toString() {
        return this.name;
    }

    static void updateVariable(Variable variable, Map<String, MathStructure> values) {
        if (values.containsKey(variable.name)) {
            MathStructure mathStructure = values.get(variable.name);
            MathStructure copy = mathStructure.createCopy();
            variable.mathStructure = copy;
        }
    }
}
