/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.rpc;

/**
 *
 * @author mmatula
 */
public interface RpcSerializable {

    public byte[] serialize();

    public void deserialize(byte[] bytes);

}
