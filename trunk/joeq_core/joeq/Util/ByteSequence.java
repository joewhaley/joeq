/*
 * ByteSequence.java
 *
 * Created on October 25, 2001, 11:17 AM
 *
 * @author  jwhaley
 * @version 
 */

package Util;

import java.io.DataInputStream;
import java.io.ByteArrayInputStream;

public final class ByteSequence extends DataInputStream {
  private ByteArrayStream byte_stream;

  public ByteSequence(byte[] bytes) { 
    super(new ByteArrayStream(bytes));
    byte_stream = (ByteArrayStream)in;
  }

  public final int getIndex()   { return byte_stream.getPosition(); }    
  final  void      unreadByte() { byte_stream.unreadByte(); }

  private static final class ByteArrayStream extends ByteArrayInputStream {
    ByteArrayStream(byte[] bytes) { super(bytes); }
    final int  getPosition() { return pos; } // is protected in ByteArrayInputStream
    final void unreadByte()  { if(pos > 0) pos--; }
  }
}
