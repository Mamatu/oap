package ogla.axischart.plugin.image;

import ogla.axischart.AxisChart;
import ogla.chart.Chart;
import java.awt.image.BufferedImage;
import javax.swing.JFrame;
import javax.swing.JPanel;

/**
 * Object of this class can deliver changes in image which is saved to file.
 * 
 */
public interface ImageComponent {

    /**
     * Set current image.
     * @param bufferedImage
     */
    public void setBufferedImage(BufferedImage bufferedImage);

    public void attachChart(AxisChart axisChartInfo);

    public void process();

    /**
     * Render image which contains changes which was delivered by this plugin.
     * @return
     */
    public abstract BufferedImage getBufferedImage();

    /**
     * Save some information about this component to byte of array.
     * This method is used by AxisChart application to save information of component
     * to file before exit of application.
     * @return array of bytes in which is stored state of object.
     */
    public abstract byte[] save();

    /**
     * Get bundle to which belong this object.
     * @return
     */
    public abstract ImageComponentBundle getBundle();

    /**
     * Get frame which can be used to configuration this component.
     * @return JFrame or null
     */
    public abstract JFrame getFrame();
}
