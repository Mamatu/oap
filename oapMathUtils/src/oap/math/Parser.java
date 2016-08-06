/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mmatula
 */
public interface Parser {

    public MathStructure[] execute(Object codeObject) throws SyntaxErrorException;

    public Object parse(String equation) throws SyntaxErrorException;

    public MathStructure[] calculate(String equation) throws SyntaxErrorException;

    public void setVariableValue(String varaible, MathStructure mathStructure);
}
