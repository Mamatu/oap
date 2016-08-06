package ogla.core.util;

import ogla.core.Help;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DefaultDocumentImpl implements Help.Document {

    private String title;
    private String content;

    private static interface Tag {

        public String getSymbol();

        public String parse(String param);
    }

    private class APTTDTag implements Tag {

        private Class clazz;

        public void setClass(Class clazz) {
            this.clazz = clazz;
        }

        public String getSymbol() {
            return "APTTD";
        }

        public String parse(String param) {
            return clazz.getResource(param).toString();
        }
    }
    private APTTDTag aPTTDTag = new APTTDTag();
    private List<Tag> tags = new ArrayList<Tag>();

    private int indexOfTag(String tagName) {
        for (int fa = 0; fa < tags.size(); fa++) {
            if (tags.get(fa).getSymbol().equals(tagName)) {
                return fa;
            }
        }
        return -1;
    }
    private StringBuilder builder = new StringBuilder();

    private class Section {

        public Section(int old, int young) {
            this.old = old;
            this.young = young;
        }
        public int old;
        public int young;
    }
    private List<Section> sections = new ArrayList<Section>();

    private String parse(String rawContent, Class clazz) {
        sections.clear();
        builder.delete(0, builder.capacity());
        aPTTDTag.setClass(clazz);
        builder.append(rawContent);
        int index = 0;
        while ((index = rawContent.indexOf("$", index)) != -1) {
            int startIndex = index;
            index++;
            StringBuilder tagName = new StringBuilder();
            StringBuilder param = new StringBuilder();
            for (; rawContent.charAt(index) != '('; index++) {
                tagName.append(rawContent.charAt(index));
            }
            int iot = indexOfTag(tagName.toString());
            if (iot != -1) {
                index++;
                for (; rawContent.charAt(index) != ')'; index++) {
                    param.append(rawContent.charAt(index));
                }
                Tag tag = tags.get(iot);
                String g = tag.parse(param.toString());
                index++;
                int nstartIndex = startIndex;
                int nindex = index;
                for (Section s : sections) {
                    nstartIndex = nstartIndex - s.old + s.young;
                    nindex = nindex - s.old + s.young;
                }

                builder.replace(nstartIndex, nindex, g);
                sections.add(new Section(index - startIndex, g.length()));
                rawContent.regionMatches(startIndex, g, 0, g.length());
            }
        }
        return builder.toString();
    }
    private StringBuilder lbuider = new StringBuilder();

    private String getContent(String path, Class clazz) {

        try {
            InputStream is = clazz.getResourceAsStream(path);
            BufferedInputStream bis = new BufferedInputStream(is);

            int i;
            while ((i = bis.read()) != -1) {
                lbuider.append((char) i);
            }

        } catch (IOException ex) {
            Logger.getLogger(DefaultDocumentImpl.class.getName()).log(Level.SEVERE, null, ex);
        }
        String code = lbuider.toString();
        lbuider.delete(0, lbuider.capacity());
        code = parse(code, clazz);
        return code;
    }

    private final void fillList() {
        tags.add(aPTTDTag);
    }

    public DefaultDocumentImpl(String title) {
        this.title = title;
        fillList();
    }

    public DefaultDocumentImpl(String title, String path, Class clazz) {
        fillList();
        this.content = getContent(path, clazz);
        this.title = title;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public void appendContent(String content) {
        this.content += content;
    }

    public void loadContent(String path, Class clazz) {
        this.content = getContent(path, clazz);
    }

    public String getTitle() {
        return title;
    }

    public String getContent() {
        return content;
    }
}
