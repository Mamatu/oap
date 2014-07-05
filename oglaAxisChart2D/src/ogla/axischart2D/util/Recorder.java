package ogla.axischart2D.util;

import ogla.axischart2D.AxisChart2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.JOptionPane;

public final class Recorder {

    private AxisChart2D axisChart;
    private Iterator<ImageWriter> iwriter = null;
    private int step = 1;

    public Recorder(AxisChart2D axisChart, int step) {
        this.axisChart = axisChart;
        this.step = step;
        if (this.step <= 0) {
            this.step = 1;
        }
    }

    public Recorder(AxisChart2D axisChart) {
        this.axisChart = axisChart;
        this.step = 1;
    }

    public void setStep(int step) {
        this.step = step;
        if (this.step <= 0) {
            this.step = 1;
        }
    }

    public void setFormat(String format) {
        Iterator<ImageWriter> iwriter = null;
        try {
            iwriter = ImageIO.getImageWritersByFormatName(format);
        } catch (IllegalArgumentException ex) {
            JOptionPane.showMessageDialog(axisChart.getChartSurface(), "This format is inaccessible.", "", JOptionPane.ERROR_MESSAGE);
            return;
        }
        this.iwriter = iwriter;
    }
    private String directory;
    private String name;

    public void setDirectory(String directory) {
        this.directory = directory;
        if (this.directory.charAt(this.directory.length() - 1) != '/') {
            this.directory = this.directory + "/";
        }
    }

    public void setName(String name) {
        this.name = name;
    }
    private int index = 0;
    private List<BufferedImage> images = new ArrayList<BufferedImage>();

    public class NoNameException extends Exception {
    }

    public class NoDirectoryException extends Exception {
    }

    public class NoFormatException extends Exception {
    }

    public BufferedImage[] getImages() {
        this.stop();
        BufferedImage[] aimages = new BufferedImage[images.size()];
        aimages = images.toArray(aimages);
        images.clear();
        return aimages;
    }

    public void save() throws NoDirectoryException, NoNameException, NoFormatException {
        if (name == null || name.equals("")) {
            throw new NoNameException();
        }
        if (directory == null || directory.equals("")) {
            throw new NoDirectoryException();
        }
        if (iwriter == null) {
            throw new NoFormatException();
        }
        this.stop();
        try {
            ImageWriter writer = (ImageWriter) iwriter.next();
            ImageOutputStream ios;
            for (int fa = 0; fa < images.size(); fa++) {
                File f = new File(directory + name + "_" + fa);
                f.createNewFile();
                ios = ImageIO.createImageOutputStream(f);
                writer.setOutput(ios);
                writer.write(images.get(fa));
                ios.close();
            }
            images.clear();
        } catch (IOException ex) {
            Logger.getLogger(Recorder.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    public void rec() {
        if (isRunned) {
            index++;
            if (index == step) {
                index = 0;
                final int width = axisChart.getChartSurface().getWidth();
                final int height = axisChart.getChartSurface().getHeight();
                BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                axisChart.getChartSurface().paint(bufferedImage.getGraphics());
                images.add(bufferedImage);
            }
        }
    }

    public boolean isRecording() {
        return isRunned;
    }

    public void start() {
        if (isRunned == false) {
            images.clear();
            index = 0;
            isRunned = true;
        }
    }
    private boolean isRunned = false;

    public void cont() {
        isRunned = true;
    }

    public void stop() {
        isRunned = false;
    }
}
