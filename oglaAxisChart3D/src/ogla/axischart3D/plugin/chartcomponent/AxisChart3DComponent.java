package ogla.axischart3D.plugin.chartcomponent;


import ogla.axischart3D.AxisChart3D;
import ogla.chart.DrawableOnChart;
import ogla.chart.DrawingInfo;
import ogla.g3d.Primitive;
import java.awt.Graphics2D;
import javax.swing.JFrame;

public abstract class AxisChart3DComponent implements DrawableOnChart {

    protected AxisChart3DComponentBundle axisChart3DComponentBundle;
    protected boolean isVisible = true;

    public AxisChart3DComponent(AxisChart3DComponentBundle axisChart3DComponentBundle) {
        this.axisChart3DComponentBundle = axisChart3DComponentBundle;
    }

    public AxisChart3DComponentBundle getBundle() {
        return axisChart3DComponentBundle;
    }

    public abstract JFrame getFrame();

    public abstract Primitive[] getPrimitives();

    public void setVisible(boolean b) {
        this.isVisible = b;
    }

    public void drawOnChart(Graphics2D gd, DrawingInfo drawingType) {
    }

    public abstract void attachTo(AxisChart3D axisChart3D);
}
