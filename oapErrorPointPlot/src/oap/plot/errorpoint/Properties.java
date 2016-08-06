
package ogla.plot.errorpoint;

import java.awt.BasicStroke;
import java.awt.Color;

public interface Properties {

    void setColor(Color color);

    void xIsPercent(boolean b);

    void yIsPercent(boolean b);

    void setXError(float[] x);

    void setYError(float[] y);

    void setXIndexOfColumnInRepository(int index);

    void setYIndexOfColumnInRepository(int index);

    void setStroke(BasicStroke stroke);
}
