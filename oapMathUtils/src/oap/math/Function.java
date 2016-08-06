package ogla.math;

import java.util.List;

/**
 *
 * @author mmatula
 */
public interface Function {

    public int getParamsCount();

    public String getName();

    public boolean rawCodeAsParams();

    public String getErrorString();

    public boolean setParams(Object params, Parser parser);

    public MathStructure execute() throws SyntaxErrorException;

}
