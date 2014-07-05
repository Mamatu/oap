/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.ui;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mmatula
 */
public abstract class DocumentationImpl implements Documentation {

    public abstract String getDescription();

    public abstract String getInfo();

    private List<Documentation> infoDeliverImpls = new ArrayList<Documentation>();

    public void addChild(Documentation child) {
        this.infoDeliverImpls.add(child);
    }

    public String getText() {
        String info = this.getInfo();
        String description = this.getDescription();
        StringBuilder builder = new StringBuilder();
        if (description.length() > 0) {
            builder.append("Description: \n");
            builder.append(description);
            builder.append("\n");
        }
        builder.append(info);
        builder.append("\n");
        return builder.toString();
    }

    public String getTextWithChildren() {
        String info = this.getInfo();
        String description = this.getDescription();
        StringBuilder builder = new StringBuilder();
        if (description.length() > 0) {
            builder.append("Description: \n");
            builder.append(description);
            builder.append("\n");
        }
        builder.append(info);
        builder.append("\n");

        for (Documentation infoDeliverImpl : infoDeliverImpls) {
            builder.append(infoDeliverImpl.getText());
        }
        return builder.toString();
    }
}
