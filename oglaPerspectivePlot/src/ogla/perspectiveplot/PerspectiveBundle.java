package ogla.perspectiveplot;

import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManagerBundle;
import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManager;
import ogla.axischart3D.plugin.projection.ProjectionBundle;
import ogla.g3d.Perspective;
import ogla.g3d.Projection;
import ogla.math.Vector3;
import javax.swing.JFrame;

public class PerspectiveBundle implements ProjectionBundle {

    private static final int GRID = 0;
    private static final int MODEL = 1;
    private Perspective perspective = new Perspective();
    private Vector3 viewerPosistion = new Vector3(0, 0, 500);
    private Frame frame = new Frame(viewerPosistion, perspective);

    @Override
    public String getLabel() {
        return "Perspective projection - default";
    }

    public String getSymbolicName() {
        return "analysis_perspective_plot_#%$##$##$";
    }

    public Projection getProjection() {
        return perspective;
    }

    public JFrame getFrame() {
        return frame;
    }
}
