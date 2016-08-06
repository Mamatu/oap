package ogla.axischart.util;

import ogla.axischart.util.ImagePanel;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JList;
import javax.swing.ListCellRenderer;

/**
 *
 * @author marcin
 */
public class ImageList extends ImagePanel implements ListCellRenderer {

    private Dimension dimension = new Dimension(200, 200);

    public static int getWidthToWhichImageIsScaled() {
        return 180;
    }

    public static int getHeightToWhichImageIsScaled() {
        return 180;
    }

    public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {

        Element element = null;
        if (value instanceof Element) {
            element = (Element) value;
        }
        if (element.label != null && element != null) {
            this.setLabel(element.label);
        }
        if (isSelected) {
            this.setBackground(Color.gray);
        } else {
            this.setBackground(Color.white);
        }
        this.setImage(element.scaledImage);
        this.setPreferredSize(dimension);
        return this;
    }

    public static class Element {

        private boolean nameIsUsed(String label) {
            for (WeakReference<Element> element : elements) {
                if (element.get() != null && element.get().label.equals(label)) {
                    return true;
                }
            }
            return false;
        }
        public static List<WeakReference<Element>> elements = new ArrayList<WeakReference<Element>>();
        public String label = "";
        private String suffix = "";
        public BufferedImage bufferedImage = null;
        public Image scaledImage = null;

        protected Element(String label) {
            if (label.length() == 0) {
                label = "chart";
            }
            for (int fa = 0; nameIsUsed(label + suffix); fa++) {
                suffix = "_" + String.valueOf(fa);
            }
            this.label = label + suffix;
            elements.add(new WeakReference(this));
        }

        public Element(String label, BufferedImage bufferedImage) {
            this(label);
            this.bufferedImage = bufferedImage;
            this.scaledImage = bufferedImage.getScaledInstance(getWidthToWhichImageIsScaled(),
                    getHeightToWhichImageIsScaled(), BufferedImage.SCALE_SMOOTH);
        }
    }
}
