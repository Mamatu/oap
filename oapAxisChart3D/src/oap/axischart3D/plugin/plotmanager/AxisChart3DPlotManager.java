package ogla.axischart3D.plugin.plotmanager;

import ogla.data.DataBlock;
import ogla.g3d.Primitive;
import ogla.g3d.Projection;
import javax.swing.JFrame;

public interface AxisChart3DPlotManager {

    public AxisChart3DPlotManagerBundle getBundle();

    public Primitive[] getPrimitives(DataBlock dataBlock);

    public JFrame getFrame();
}
