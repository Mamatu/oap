package ogla.axischart3D.plugin.projection;

import ogla.g3d.Projection;
import javax.swing.JFrame;

public interface ProjectionBundle {

    public Projection getProjection();

    public String getLabel();

    public String getSymbolicName();

    public JFrame getFrame();
}
