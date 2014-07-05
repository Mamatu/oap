package ogla.axischart.plugin.exporter;

import java.awt.image.BufferedImage;
import javax.imageio.ImageWriter;
import javax.swing.JFrame;
import javax.swing.JPanel;

public interface ImageExporter {

    public void setImage(BufferedImage bufferedImage);

    public BufferedImage getImage();

    public JFrame getFrame();

    public ImageWriter getImageWriter();

    public BundleImageExporter getBundle();

    public String getDescription();

    public String getSuffix();

    public boolean accept(String name);
}
