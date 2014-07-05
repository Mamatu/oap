package ogla.label;

import ogla.excore.util.Writer;
import ogla.axischart.AxisChart;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.axischart.plugin.image.ImageComponent;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import org.scilab.forge.jlatexmath.TeXIcon;
import org.scilab.forge.jlatexmath.TeXConstants;
import org.scilab.forge.jlatexmath.TeXFormula;

public class Label implements ImageComponent {

    public static final int CENTER_ALIGN = 0;
    public static final int LEFT_ALIGN = 1;
    public static final int RIGHT_ALIGN = 2;

    private class Properties {

        public int align = Label.CENTER_ALIGN;
    }
    private Properties properties = new Properties();
    private BundleLabel bundleLabel;
    private BufferedImage currentBufferedImage = null;
    private Color color = null;

    public Label(BundleLabel bundleLabel) {
        this.bundleLabel = bundleLabel;
    }
    private ConfigurationFrame configurationFrame = new ConfigurationFrame();
    private AxisChart axisChart = null;

    @Override
    public void attachChart(AxisChart axisChart) {
        color = axisChart.getChartSurface().getBackground();
        this.axisChart = axisChart;
    }

    public void setTop(String s) {
        configurationFrame.setTop(s);
    }

    public String getTop() {
        return configurationFrame.getTop();
    }

    public void setBottom(String s) {
        configurationFrame.setBottom(s);
    }

    public String getBottom() {
        return configurationFrame.getBottom();
    }

    public void setRight(String s) {
        configurationFrame.setRight(s);
    }

    public String getRight() {
        return configurationFrame.getRight();
    }

    public void setLeft(String s) {
        configurationFrame.setLeft(s);
    }

    public String getLeft() {
        return configurationFrame.getLeft();
    }

    public void process() {
    }

    @Override
    public void setBufferedImage(BufferedImage bufferedImage) {
        currentBufferedImage = bufferedImage;
    }

    @Override
    public JFrame getFrame() {
        return configurationFrame;
    }

    private int numberOfLines(String text) {
        int i = 0;
        int count = 0;
        while ((i = text.indexOf("\n", i)) != -1) {
            count++;
            i++;
        }
        return count;
    }

    public void drawString(String text, Graphics2D gd, int x, int y, BufferedImage bufferedImage) {
        if (text == null || text.length() == 0) {
            return;
        }
        String[] lines = text.split("\n");
        TeXIcon prev = null;
        int prevY = 0;
        for (int fa = 0; fa < lines.length; fa++) {

            TeXFormula formule = new TeXFormula(lines[fa]);

            TeXIcon ti = formule.createTeXIcon(
                    TeXConstants.STYLE_DISPLAY, configurationFrame.getFontSize());
            int yextra = prevY;
            int xextra = 0;
            if (prev != null) {
                yextra += ti.getTrueIconHeight() + 2;
            }
            if (properties.align == Label.CENTER_ALIGN) {
                xextra = bufferedImage.getWidth() / 2;
            } else if (properties.align == Label.LEFT_ALIGN) {
                xextra = 0;
            } else if (properties.align == Label.RIGHT_ALIGN) {
                xextra = bufferedImage.getWidth();
            }
            ti.paintIcon(this.axisChart.getChartSurface(), gd, x + xextra, y + yextra);
            prevY += yextra;
            prev = ti;
        }
    }

    public void drawStringPerpendicular(String text, Graphics2D gd, int x, int y, BufferedImage bufferedImage) {
        if (text == null || text.length() == 0) {
            return;
        }
        String[] lines = text.split("\n");
        TeXIcon prev = null;
        int prevX = 0;
        for (int fa = 0; fa < lines.length; fa++) {
            int length = getLengthOfText(lines[fa], gd);

            TeXFormula fomule = new TeXFormula(lines[fa]);

            TeXIcon ti = fomule.createTeXIcon(
                    TeXConstants.STYLE_DISPLAY, configurationFrame.getFontSize());
            int xextra = prevX;
            int yextra = 0;
            if (prev != null) {
                xextra += ti.getTrueIconHeight() + 2;
            }
            if (properties.align == Label.CENTER_ALIGN) {
                yextra = bufferedImage.getHeight() / 2;
            } else if (properties.align == Label.LEFT_ALIGN) {
                yextra = 0;
            } else if (properties.align == Label.RIGHT_ALIGN) {
                yextra = bufferedImage.getHeight();
            }
            gd.rotate(-Math.PI / 2.f, x + xextra, y + yextra);
            ti.paintIcon(this.axisChart.getChartSurface(), gd, x + xextra, y + yextra);
            gd.rotate(Math.PI / 2.f, x + xextra, y + yextra);
            prevX = x + xextra;
            prev = ti;
        }
    }

    public void drawStringPerpendicular1(String text, Graphics2D gd, int x, int y, BufferedImage bufferedImage) {
        if (text == null || text.length() == 0) {
            return;
        }
        String[] lines = text.split("\n");
        TeXIcon prev = null;
        int prevX = 0;
        for (int fa = 0; fa < lines.length; fa++) {
            int length = getLengthOfText(lines[fa], gd);

            TeXFormula fomule = new TeXFormula(lines[fa]);
            TeXIcon ti = fomule.createTeXIcon(
                    TeXConstants.STYLE_DISPLAY, configurationFrame.getFontSize());
            int xextra = prevX;
            int yextra = 0;
            if (prev != null) {
                xextra -= ti.getTrueIconHeight() - 2;
            }
            if (properties.align == Label.CENTER_ALIGN) {
                yextra = bufferedImage.getHeight() / 2;
            } else if (properties.align == Label.LEFT_ALIGN) {
                yextra = 0;
            } else if (properties.align == Label.RIGHT_ALIGN) {
                yextra = bufferedImage.getHeight();
            }
            gd.rotate(Math.PI / 2.f, x + xextra, y + yextra);
            ti.paintIcon(this.axisChart.getChartSurface(), gd, x + xextra, y + yextra);
            gd.rotate(-Math.PI / 2.f, x + xextra, y + yextra);
            prevX += xextra;
            prev = ti;
        }
    }

    public int getLengthOfText(String text, Graphics2D gd) {
        int length = 0;
        for (int fa = 0; fa < text.length(); fa++) {
            length += gd.getFontMetrics().charWidth(text.charAt(fa));
        }
        return length;
    }

    class Icon {
    }

    @Override
    public BufferedImage getBufferedImage() {
        String topX = configurationFrame.getTop();
        int nTop = numberOfLines(topX);

        String bottomX = configurationFrame.getBottom();
        int nBottom = numberOfLines(bottomX);

        String rightY = configurationFrame.getRight();
        int nRight = numberOfLines(rightY);

        String leftY = configurationFrame.getLeft();
        int nLeft = numberOfLines(leftY);

        int sizeOfFont = currentBufferedImage.getGraphics().getFont().getSize();

        BufferedImage bufferedImage = new BufferedImage(
                currentBufferedImage.getWidth() + nRight * sizeOfFont + nLeft * sizeOfFont + 3,
                currentBufferedImage.getHeight() + nTop * sizeOfFont + nBottom * sizeOfFont + 3,
                BufferedImage.TYPE_INT_RGB);

        Graphics2D gd = (Graphics2D) bufferedImage.getGraphics();
        Font defaultFont = gd.getFont();
        if (color == null) {
            color = new Color(240, 240, 240);
        }
        gd.setBackground(color);
        gd.clearRect(0, 0, currentBufferedImage.getWidth() + nRight * (sizeOfFont + 2) + nLeft * (sizeOfFont + 2),
                currentBufferedImage.getHeight() + nTop * (sizeOfFont + 2) + nBottom * (sizeOfFont + 2));
        gd.drawImage(currentBufferedImage, null, nLeft * (sizeOfFont + 2), nTop * (sizeOfFont + 2));
        gd.setColor(Color.black);
        TeXFormula.registerExternalFont(Character.UnicodeBlock.BASIC_LATIN,
                "arial");
        drawString(topX, gd, 0, 0, bufferedImage);
        drawString(bottomX, gd, 0,
                currentBufferedImage.getHeight(), bufferedImage);
        drawStringPerpendicular1(rightY, gd, bufferedImage.getWidth(),
                0, bufferedImage);
        drawStringPerpendicular(leftY, gd, 0, 0, bufferedImage);
        gd.setFont(defaultFont);
        configurationFrame.clearBuffers();
        return bufferedImage;
    }

    @Override
    public byte[] save() {
        Writer writer = new Writer();
        try {
            writer.write(configurationFrame.getTop().length());
            writer.write(configurationFrame.getTop());
            writer.write(configurationFrame.getBottom().length());
            writer.write(configurationFrame.getBottom());
            writer.write(configurationFrame.getRight().length());
            writer.write(configurationFrame.getRight());
            writer.write(configurationFrame.getLeft().length());
            writer.write(configurationFrame.getLeft());
        } catch (IOException ex) {
            Logger.getLogger(Label.class.getName()).log(Level.SEVERE, null, ex);
        }
        return writer.getBytes();
    }

    @Override
    public ImageComponentBundle getBundle() {
        return bundleLabel;
    }
}
