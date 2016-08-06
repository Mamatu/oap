/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.ui;

import ogla.core.rpc.FunctionInfo;
import ogla.core.util.ArgumentType;

public abstract class OglaMethod extends FunctionInfo implements Documentation {

    private MethodInfo methodInfo = null;

    protected OglaMethod() {
        super(null);
    }

    public OglaMethod(String name, ArgumentType[] inargs) {
        super(null);
        this.setNames(new String[]{name});
        this.setInputArgs(inargs);
        methodInfo = this.new MethodInfo();
    }

    public OglaMethod(String[] names, ArgumentType[] inargs) {
        super(null);
        this.setNames(names);
        this.setInputArgs(inargs);
        methodInfo = this.new MethodInfo();
    }

    public OglaMethod(String[] names, ArgumentType[] inargs, ArgumentType[] outargs) {
        super(null);
        this.setNames(names);
        this.setInputArgs(inargs);
        this.setOutputArgs(outargs);
        methodInfo = this.new MethodInfo();
    }

    /**
     *
     * @param args
     * @param oglaObject
     * @param userData
     * @return
     */
    public abstract Object invoke(Object args, OglaObject oglaObject, Object userData);

    public String getArgumentsInfo(ArgumentType[] types) {
        if (types == null) {
            return "";
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (int fa = 0; fa < types.length - 1; fa++) {
            stringBuilder.append(ArgumentType.toString(types[fa]));
            stringBuilder.append(", ");
        }
        stringBuilder.append(ArgumentType.toString(types[types.length - 1]));
        return stringBuilder.toString();
    }

    public String getInputArgumentsInfo() {
        return this.getArgumentsInfo(this.getInputArgsRef());
    }

    public String getOutputArgumentsInfo() {
        return this.getArgumentsInfo(this.getOutputArgsRef());
    }

    @Override
    public String getText() {
        return this.methodInfo.getText();
    }

    @Override
    public String getTextWithChildren() {
        return this.methodInfo.getTextWithChildren();
    }

    private class MethodInfo extends DocumentationImpl {

        @Override
        public String getInfo() {
            String input = OglaMethod.this.getInputArgumentsInfo();
            String output = OglaMethod.this.getOutputArgumentsInfo();
            String description = this.getDescription();
            StringBuilder builder = new StringBuilder();
            builder.append("   ");
            builder.append(output);
            builder.append(" ");
            builder.append(OglaMethod.this.getName());
            builder.append("(");
            builder.append(input);
            builder.append(")");
            return builder.toString();
        }

        @Override
        public String getDescription() {
            return "";
        }
    }
}
