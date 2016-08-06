/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.rpc;

import ogla.core.util.ArgumentType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 *
 * @author mmatula
 */
public abstract class RpcFunction extends FunctionInfo {

    public RpcFunction() {
        super(null);
    }
    
    public RpcFunction(String name, ArgumentType[] inargs, ArgumentType[] outargs) {
        super(null);
        this.setName(name);
        this.setInputArgs(inargs);
        this.setOutputArgs(outargs);
    }
    
    public abstract ByteBuffer invoke(ByteBuffer input);
}
