package ogla.polar;

import ogla.excore.Help;
import ogla.excore.util.Reader;
import ogla.excore.util.Reader.EndOfBufferException;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponentBundle;
import java.awt.Color;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class BundlePolarGrid implements AxisChart2DComponentBundle {

    private int index = 0;

    public int getIndex() {
        int out = index;
        index++;
        return out;
    }

    public AxisChart2DComponent newAxisChart2DComponent() {
        return new PolarGrid(this, getIndex());
    }

    public AxisChart2DComponent load(byte[] bytes) {
        Reader reader = new Reader(bytes);
        PolarGrid polarGrid = new PolarGrid(this, getIndex());
        try {
            int angleToSave = reader.readInt();
            polarGrid.setAngleOfLines((int) angleToSave);
            int gap = reader.readInt();
            polarGrid.gap = gap;
            int gridColor = reader.readInt();
            polarGrid.setGridColor(new Color(gridColor));
            int textColor = reader.readInt();
            polarGrid.setTextColor(new Color(textColor));
            boolean linesAreVisible = reader.readBoolean();
            boolean gridIsVisible = reader.readBoolean();
            boolean textIsVisible = reader.readBoolean();
            polarGrid.linesAreVisible = linesAreVisible;
            polarGrid.gridIsVisible = gridIsVisible;
            polarGrid.textIsVisible = textIsVisible;

            int type = reader.readInt();
            if (type == 0) {
                polarGrid.isLeftCorner = true;
            } else if (type == 1) {
                float tickx = reader.readFloat();
                float ticky = reader.readFloat();
                polarGrid.isTick = true;
                polarGrid.tickx = new Float(tickx);
                polarGrid.ticky = new Float(ticky);
            } else if (type == 2) {
                int pixelx = reader.readInt();
                int pixely = reader.readInt();
                polarGrid.isPixel = true;
                polarGrid.pixelx = pixelx;
                polarGrid.pixely = pixely;
            }
            int usedAxis = reader.readInt();
            polarGrid.usedAxis = usedAxis;
        } catch (EndOfBufferException ex) {
            Logger.getLogger(BundlePolarGrid.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(BundlePolarGrid.class.getName()).log(Level.SEVERE, null, ex);
        }
        return polarGrid;
    }

    public boolean canBeSaved() {
        return true;
    }

    public String getSymbolicName() {
        return "analysis_polar_grid_&*^^%(*)^%^&^";
    }

    public Help getHelp() {
        return null;
    }

    public String getLabel() {
        return "Polar grid";
    }

    public String getDescription() {
        return "";
    }
}
