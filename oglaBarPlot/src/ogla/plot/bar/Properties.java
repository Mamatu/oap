/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.plot.bar;

import java.awt.BasicStroke;
import java.awt.Color;

/**
 *
 * @author marcin
 */
public interface Properties {

    void setColorOfFilling(Color color);

    void setColorOfEdge(Color color);

    void setStroke(BasicStroke Stroke);

    void setOriginValue(Float originValue);

    void setOriginIndex(Integer originIndex);
}
