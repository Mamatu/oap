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
public interface Operator {

    public int getWeight();

    public char getSymbol();

    public boolean setParams(MathStructure param1, MathStructure param2);

    /**
     *
     * @param param1 first parameter and output of calculations
     * @param param2 second parameter
     * @return
     * @throws SyntaxErrorException
     */
    public MathStructure execute() throws SyntaxErrorException;
}
