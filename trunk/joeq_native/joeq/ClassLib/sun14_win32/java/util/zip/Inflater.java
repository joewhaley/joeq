/* java.util.zip.Inflater
   Copyright (C) 2001 Free Software Foundation, Inc.

This file is part of GNU Classpath.

GNU Classpath is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GNU Classpath is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Classpath; see the file COPYING.  If not, write to the
Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
02111-1307 USA.

Linking this library statically or dynamically with other modules is
making a combined work based on this library.  Thus, the terms and
conditions of the GNU General Public License cover the whole
combination.

As a special exception, the copyright holders of this library give you
permission to link this library with independent modules to produce an
executable, regardless of the license terms of these independent
modules, and to copy and distribute the resulting executable under
terms of your choice, provided that you also meet, for each linked
independent module, the terms and conditions of the license of that
module.  An independent module is a module which is not derived from
or based on this library.  If you modify this library, you may extend
this exception to your version of the library, but you are not
obligated to do so.  If you do not wish to do so, delete this
exception statement from your version. */

package ClassLib.sun14_win32.java.util.zip;
import Clazz.*;
import Bootstrap.*;
import Run_Time.Reflection;

/**
 * Inflater is used to decompress data that has been compressed according 
 * to the "deflate" standard described in rfc1950.
 *
 * The usage is as following.  First you have to set some input with
 * <code>setInput()</code>, then inflate() it.  If inflate doesn't
 * inflate any bytes there may be three reasons:
 * <ul>
 * <li>needsInput() returns true because the input buffer is empty.
 * You have to provide more input with <code>setInput()</code>.  
 * NOTE: needsInput() also returns true when, the stream is finished.
 * </li>
 * <li>needsDictionary() returns true, you have to provide a preset 
 *     dictionary with <code>setDictionary()</code>.</li>
 * <li>finished() returns true, the inflater has finished.</li>
 * </ul>
 * Once the first output byte is produced, a dictionary will not be
 * needed at a later stage.
 *
 * @author John Leuner, Jochen Hoenicke
 * @since JDK 1.1
 */
public class Inflater
{
  /* Copy lengths for literal codes 257..285 */
  private static final int CPLENS[] = 
  { 
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
  };
  
  /* Extra bits for literal codes 257..285 */  
  private static final int CPLEXT[] = 
  { 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
  };

  /* Copy offsets for distance codes 0..29 */
  private static final int CPDIST[] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577
  };
  
  /* Extra bits for distance codes */
  private static final int CPDEXT[] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 
    12, 12, 13, 13
  };

  /* This are the state in which the inflater can be.  */
  private static final int DECODE_HEADER           = 0;
  private static final int DECODE_DICT             = 1;
  private static final int DECODE_BLOCKS           = 2;
  private static final int DECODE_STORED_LEN1      = 3;
  private static final int DECODE_STORED_LEN2      = 4;
  private static final int DECODE_STORED           = 5;
  private static final int DECODE_DYN_HEADER       = 6;
  private static final int DECODE_HUFFMAN          = 7;
  private static final int DECODE_HUFFMAN_LENBITS  = 8;
  private static final int DECODE_HUFFMAN_DIST     = 9;
  private static final int DECODE_HUFFMAN_DISTBITS = 10;
  private static final int DECODE_CHKSUM           = 11;
  private static final int FINISHED                = 12;

  /** This variable contains the current state. */
  private int mode;

  /**
   * The adler checksum of the dictionary or of the decompressed
   * stream, as it is written in the header resp. footer of the
   * compressed stream.  <br>
   *
   * Only valid if mode is DECODE_DICT or DECODE_CHKSUM.
   */
  private int readAdler;
  /** 
   * The number of bits needed to complete the current state.  This
   * is valid, if mode is DECODE_DICT, DECODE_CHKSUM,
   * DECODE_HUFFMAN_LENBITS or DECODE_HUFFMAN_DISTBITS.  
   */
  private int neededBits;
  private int repLength, repDist;
  private int uncomprLen;
  /**
   * True, if the last block flag was set in the last block of the
   * inflated stream.  This means that the stream ends after the
   * current block.  
   */
  private boolean isLastBlock;

  /**
   * The total number of inflated bytes.
   */
  private int totalOut;
  /**
   * The total number of bytes set with setInput().  This is not the
   * value returned by getTotalIn(), since this also includes the 
   * unprocessed input.
   */
  private int totalIn;
  /**
   * This variable stores the nowrap flag that was given to the constructor.
   * True means, that the inflated stream doesn't contain a header nor the
   * checksum in the footer.
   */
  private boolean nowrap;

  private StreamManipulator input;
  private OutputWindow outputWindow;
  private InflaterDynHeader dynHeader;
  private InflaterHuffmanTree litlenTree, distTree;
  private java.util.zip.Adler32 adler;

  /**
   * Creates a new inflater.
   * @param nowrap true if no header and checksum field appears in the
   * stream.  This is used for GZIPed input.  For compatibility with
   * Sun JDK you should provide one byte of input more than needed in
   * this case.
   */
  public static void __init__(java.util.zip.Inflater dis, boolean nowrap) 
  {
      System.out.println("Init: "+dis);
      Reflection.putfield_Z(dis, _nowrap, nowrap);
      Reflection.putfield_A(dis, _adler, new java.util.zip.Adler32());
      Reflection.putfield_A(dis, _input, new StreamManipulator());
      Reflection.putfield_A(dis, _outputWindow, new OutputWindow());
      Reflection.putfield_I(dis, _mode, nowrap ? DECODE_BLOCKS : DECODE_HEADER);
  }

  /**
   * Resets the inflater so that a new stream can be decompressed.  All
   * pending input and output will be discarded.
   */
  public static void reset(java.util.zip.Inflater dis)
  {
      boolean nowrap = Reflection.getfield_Z(dis, _nowrap);
      Reflection.putfield_I(dis, _mode, nowrap ? DECODE_BLOCKS : DECODE_HEADER);
      Reflection.putfield_I(dis, _totalIn, 0);
      Reflection.putfield_I(dis, _totalOut, 0);
      ((StreamManipulator)Reflection.getfield_A(dis, _input)).reset();
      ((OutputWindow)Reflection.getfield_A(dis, _outputWindow)).reset();
      Reflection.putfield_A(dis, _dynHeader, null);
      Reflection.putfield_A(dis, _litlenTree, null);
      Reflection.putfield_A(dis, _distTree, null);
      Reflection.putfield_Z(dis, _isLastBlock, false);
      ((java.util.zip.Adler32)Reflection.getfield_A(dis, _adler)).reset();
  }

  /**
   * Decodes the deflate header.
   * @return false if more input is needed. 
   * @exception DataFormatException if header is invalid.
   */
  private static boolean decodeHeader(java.util.zip.Inflater dis) throws java.util.zip.DataFormatException
  {
    StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
    int header = input.peekBits(16);
    if (header < 0)
      return false;
    input.dropBits(16);
    
    /* The header is written in "wrong" byte order */
    header = ((header << 8) | (header >> 8)) & 0xffff;
    if (header % 31 != 0)
      throw new java.util.zip.DataFormatException("Header checksum illegal");
    
    if ((header & 0x0f00) != (java.util.zip.Deflater.DEFLATED << 8))
      throw new java.util.zip.DataFormatException("Compression Method unknown");

    /* Maximum size of the backwards window in bits. 
     * We currently ignore this, but we could use it to make the
     * inflater window more space efficient. On the other hand the
     * full window (15 bits) is needed most times, anyway.
     int max_wbits = ((header & 0x7000) >> 12) + 8;
     */
    
    if ((header & 0x0020) == 0) // Dictionary flag?
      {
	  Reflection.putfield_I(dis, _mode, DECODE_BLOCKS);
      }
    else
      {
	  Reflection.putfield_I(dis, _mode, DECODE_DICT);
	  Reflection.putfield_I(dis, _neededBits, 32);
      }
    return true;
  }
   
  /**
   * Decodes the dictionary checksum after the deflate header.
   * @return false if more input is needed. 
   */
  private static boolean decodeDict(java.util.zip.Inflater dis)
  {
      for (;;) {
	  int neededBits = Reflection.getfield_I(dis, _neededBits);
	  if (neededBits <= 0) break;
	  StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
	  int dictByte = input.peekBits(8);
	  if (dictByte < 0)
	      return false;
	  input.dropBits(8);
	  int readAdler = Reflection.getfield_I(dis, _readAdler);
	  Reflection.putfield_I(dis, _readAdler, (readAdler << 8) | dictByte);
	  Reflection.putfield_I(dis, _neededBits, neededBits - 8);
      }
      return false;
  }

  /**
   * Decodes the huffman encoded symbols in the input stream.
   * @return false if more input is needed, true if output window is
   * full or the current block ends.
   * @exception DataFormatException if deflated stream is invalid.  
   */
  private static boolean decodeHuffman(java.util.zip.Inflater dis) throws java.util.zip.DataFormatException
  {
      OutputWindow outputWindow = (OutputWindow)Reflection.getfield_A(dis, _outputWindow);
      StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
    int free = outputWindow.getFreeSpace();
    while (free >= 258)
      {
	int symbol;
	int mode = Reflection.getfield_I(dis, _mode);
	switch (mode)
	  {
	  case DECODE_HUFFMAN:
	    /* This is the inner loop so it is optimized a bit */
	      InflaterHuffmanTree litlenTree = (InflaterHuffmanTree)Reflection.getfield_A(dis, _litlenTree);
	    while (((symbol = litlenTree.getSymbol(input)) & ~0xff) == 0)
	      {
		outputWindow.write(symbol);
		if (--free < 258)
		  return true;
	      } 
	    if (symbol < 257)
	      {
		if (symbol < 0)
		  return false;
		else
		  {
		    /* symbol == 256: end of block */
		      Reflection.putfield_A(dis, _distTree, null);
		      Reflection.putfield_A(dis, _litlenTree, null);
		      Reflection.putfield_I(dis, _mode, DECODE_BLOCKS);
		    return true;
		  }
	      }
		
	    try
	      {
		  Reflection.putfield_I(dis, _repLength, CPLENS[symbol - 257]);
		  Reflection.putfield_I(dis, _neededBits, CPLEXT[symbol - 257]);
	      }
	    catch (ArrayIndexOutOfBoundsException ex)
	      {
		throw new java.util.zip.DataFormatException("Illegal rep length code");
	      }
	    /* fall through */
	  case DECODE_HUFFMAN_LENBITS:
	      int neededBits = Reflection.getfield_I(dis, _neededBits);
	    if (neededBits > 0)
	      {
		  Reflection.putfield_I(dis, _mode, DECODE_HUFFMAN_LENBITS);
		int i = input.peekBits(neededBits);
		if (i < 0)
		  return false;
		input.dropBits(neededBits);
		int repLength = Reflection.getfield_I(dis, _repLength);
		Reflection.putfield_I(dis, _repLength, repLength + i);
	      }
	    Reflection.putfield_I(dis, _mode, DECODE_HUFFMAN_DIST);
	    /* fall through */
	  case DECODE_HUFFMAN_DIST:
	      InflaterHuffmanTree distTree = (InflaterHuffmanTree)Reflection.getfield_A(dis, _distTree);
	    symbol = distTree.getSymbol(input);
	    if (symbol < 0)
	      return false;
	    try 
	      {
		  Reflection.putfield_I(dis, _repDist, CPDIST[symbol]);
		  Reflection.putfield_I(dis, _neededBits, CPDEXT[symbol]);
	      }
	    catch (ArrayIndexOutOfBoundsException ex)
	      {
		throw new java.util.zip.DataFormatException("Illegal rep dist code");
	      }
	    /* fall through */
	  case DECODE_HUFFMAN_DISTBITS:
	      neededBits = Reflection.getfield_I(dis, _neededBits);

	    if (neededBits > 0)
	      {
		  Reflection.putfield_I(dis, _mode, DECODE_HUFFMAN_DISTBITS);
		int i = input.peekBits(neededBits);
		if (i < 0)
		  return false;
		input.dropBits(neededBits);
		int repDist = Reflection.getfield_I(dis, _repDist);
		Reflection.putfield_I(dis, _repDist, repDist + i);
	      }
	    int repLength = Reflection.getfield_I(dis, _repLength);
	    int repDist = Reflection.getfield_I(dis, _repDist);
	    outputWindow.repeat(repLength, repDist);
	    free -= repLength;
	    Reflection.putfield_I(dis, _mode, DECODE_HUFFMAN);
	    break;
	  default:
	    throw new IllegalStateException();
	  }
      }
    return true;
  }

  /**
   * Decodes the adler checksum after the deflate stream.
   * @return false if more input is needed. 
   * @exception DataFormatException if checksum doesn't match.
   */
  private static boolean decodeChksum(java.util.zip.Inflater dis) throws java.util.zip.DataFormatException
  {
      for (;;) {
	  int neededBits = Reflection.getfield_I(dis, _neededBits);
	  if (neededBits <= 0) break;
	  StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
	  int chkByte = input.peekBits(8);
	  if (chkByte < 0)
	      return false;
	  input.dropBits(8);
	  int readAdler = Reflection.getfield_I(dis, _readAdler);
	  Reflection.putfield_I(dis, _readAdler, (readAdler << 8) | chkByte);
	  Reflection.putfield_I(dis, _neededBits, neededBits - 8);
      }
      java.util.zip.Adler32 adler = (java.util.zip.Adler32)Reflection.getfield_A(dis, _adler);
      int readAdler = Reflection.getfield_I(dis, _readAdler);
      if ((int) adler.getValue() != readAdler)
	  throw new java.util.zip.DataFormatException("Adler chksum doesn't match: "
						      +Integer.toHexString((int)adler.getValue())
						      +" vs. "+Integer.toHexString(readAdler));
      
      Reflection.putfield_I(dis, _mode, FINISHED);
      return false;
  }

  /**
   * Decodes the deflated stream.
   * @return false if more input is needed, or if finished. 
   * @exception DataFormatException if deflated stream is invalid.
   */
  private static boolean decode(java.util.zip.Inflater dis) throws java.util.zip.DataFormatException
  {
      StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
      int mode = Reflection.getfield_I(dis, _mode);
    switch (mode) 
      {
      case DECODE_HEADER:
	return decodeHeader(dis);
      case DECODE_DICT:
	return decodeDict(dis);
      case DECODE_CHKSUM:
	return decodeChksum(dis);

      case DECODE_BLOCKS:
	  boolean isLastBlock = Reflection.getfield_Z(dis, _isLastBlock);
	if (isLastBlock)
	  {
	      boolean nowrap = Reflection.getfield_Z(dis, _nowrap);
	    if (nowrap)
	      {
		  Reflection.putfield_I(dis, _mode, FINISHED);
		return false;
	      }
	    else
	      {
		input.skipToByteBoundary();
		Reflection.putfield_I(dis, _neededBits, 32);
		  Reflection.putfield_I(dis, _mode, DECODE_CHKSUM);
		return true;
	      }
	  }

	int type = input.peekBits(3);
	if (type < 0)
	  return false;
	input.dropBits(3);

	if ((type & 1) != 0)
	    Reflection.putfield_Z(dis, _isLastBlock, true);
	switch (type >> 1)
	  {
	  case DeflaterConstants.STORED_BLOCK:
	    input.skipToByteBoundary();
	    Reflection.putfield_I(dis, _mode, DECODE_STORED_LEN1);
	    break;
	  case DeflaterConstants.STATIC_TREES:
	    Reflection.putfield_A(dis, _litlenTree, InflaterHuffmanTree.defLitLenTree);
            Reflection.putfield_A(dis, _distTree, InflaterHuffmanTree.defDistTree);
	    Reflection.putfield_I(dis, _mode, DECODE_HUFFMAN);
	    break;
	  case DeflaterConstants.DYN_TREES:
            Reflection.putfield_A(dis, _dynHeader, new InflaterDynHeader());
	    Reflection.putfield_I(dis, _mode, DECODE_DYN_HEADER);
	    break;
	  default:
	    throw new java.util.zip.DataFormatException("Unknown block type "+type);
	  }
	return true;

      case DECODE_STORED_LEN1:
	{
	    int uncomprLen = input.peekBits(16);
	    Reflection.putfield_I(dis, _uncomprLen, uncomprLen);
	    if (uncomprLen < 0)
		return false;
	    input.dropBits(16);
	    Reflection.putfield_I(dis, _mode, DECODE_STORED_LEN2);
	}
	/* fall through */
      case DECODE_STORED_LEN2:
	{
	  int nlen = input.peekBits(16);
	  if (nlen < 0)
	    return false;
	  input.dropBits(16);
	    int uncomprLen = Reflection.getfield_I(dis, _uncomprLen);
	  if (nlen != (uncomprLen ^ 0xffff))
	    throw new java.util.zip.DataFormatException("broken uncompressed block");
	  Reflection.putfield_I(dis, _mode, DECODE_STORED);
	}
	/* fall through */
      case DECODE_STORED:
	{
	  int uncomprLen = Reflection.getfield_I(dis, _uncomprLen);
	  OutputWindow outputWindow = (OutputWindow)Reflection.getfield_A(dis, _outputWindow);
	  int more = outputWindow.copyStored(input, uncomprLen);
	  Reflection.putfield_I(dis, _uncomprLen, uncomprLen -= more);
	  if (uncomprLen == 0)
	    {
	      Reflection.putfield_I(dis, _mode, DECODE_BLOCKS);
	      return true;
	    }
	  return !input.needsInput();
	}

      case DECODE_DYN_HEADER:
	  InflaterDynHeader dynHeader = (InflaterDynHeader)Reflection.getfield_A(dis, _dynHeader);
	  if (!dynHeader.decode(input))
	      return false;
	  Reflection.putfield_A(dis, _litlenTree, dynHeader.buildLitLenTree());
	  Reflection.putfield_A(dis, _distTree, dynHeader.buildDistTree());
	  Reflection.putfield_I(dis, _mode, DECODE_HUFFMAN);
	/* fall through */
      case DECODE_HUFFMAN:
      case DECODE_HUFFMAN_LENBITS:
      case DECODE_HUFFMAN_DIST:
      case DECODE_HUFFMAN_DISTBITS:
	return decodeHuffman(dis);
      case FINISHED:
	return false;
      default:
	throw new IllegalStateException();
      }	
  }

  /**
   * Sets the preset dictionary.  This should only be called, if
   * needsDictionary() returns true and it should set the same
   * dictionary, that was used for deflating.  The getAdler()
   * function returns the checksum of the dictionary needed.
   * @param buffer the dictionary.
   * @exception IllegalStateException if no dictionary is needed.
   * @exception IllegalArgumentException if the dictionary checksum is
   * wrong.  
   */
  public static void setDictionary(java.util.zip.Inflater dis, byte[] buffer)
  {
    setDictionary(dis, buffer, 0, buffer.length);
  }

  /**
   * Sets the preset dictionary.  This should only be called, if
   * needsDictionary() returns true and it should set the same
   * dictionary, that was used for deflating.  The getAdler()
   * function returns the checksum of the dictionary needed.
   * @param buffer the dictionary.
   * @param off the offset into buffer where the dictionary starts.
   * @param len the length of the dictionary.
   * @exception IllegalStateException if no dictionary is needed.
   * @exception IllegalArgumentException if the dictionary checksum is
   * wrong.  
   * @exception IndexOutOfBoundsException if the off and/or len are wrong.
   */
  public static void setDictionary(java.util.zip.Inflater dis, byte[] buffer, int off, int len)
  {
    if (!needsDictionary(dis))
      throw new IllegalStateException();

    java.util.zip.Adler32 adler = (java.util.zip.Adler32)Reflection.getfield_A(dis, _adler);
    adler.update(buffer, off, len);
    int readAdler = Reflection.getfield_I(dis, _readAdler);
    if ((int) adler.getValue() != readAdler)
      throw new IllegalArgumentException("Wrong adler checksum");
    adler.reset();
    OutputWindow outputWindow = (OutputWindow)Reflection.getfield_A(dis, _outputWindow);
    outputWindow.copyDict(buffer, off, len);
    Reflection.putfield_I(dis, _mode, DECODE_BLOCKS);
  }

  /**
   * Sets the input.  This should only be called, if needsInput()
   * returns true.
   * @param buffer the input.
   * @exception IllegalStateException if no input is needed.
   */
  public static void setInput(java.util.zip.Inflater dis, byte[] buf) 
  {
    setInput(dis, buf, 0, buf.length);
  }

  /**
   * Sets the input.  This should only be called, if needsInput()
   * returns true.
   * @param buffer the input.
   * @param off the offset into buffer where the input starts.
   * @param len the length of the input.  
   * @exception IllegalStateException if no input is needed.
   * @exception IndexOutOfBoundsException if the off and/or len are wrong.
   */
  public static void setInput(java.util.zip.Inflater dis, byte[] buf, int off, int len) 
  {
      StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
    input.setInput(buf, off, len);
    int totalIn = Reflection.getfield_I(dis, _totalIn);
    Reflection.putfield_I(dis, _totalIn, totalIn + len);
  }

  /**
   * Inflates the compressed stream to the output buffer.  If this
   * returns 0, you should check, whether needsDictionary(),
   * needsInput() or finished() returns true, to determine why no 
   * further output is produced.
   * @param buffer the output buffer.
   * @return the number of bytes written to the buffer, 0 if no further
   * output can be produced.  
   * @exception DataFormatException if deflated stream is invalid.
   * @exception IllegalArgumentException if buf has length 0.
   */
  public static int inflate(java.util.zip.Inflater dis, byte[] buf) throws java.util.zip.DataFormatException
  {
    return inflate(dis, buf, 0, buf.length);
  }
  
  /**
   * Inflates the compressed stream to the output buffer.  If this
   * returns 0, you should check, whether needsDictionary(),
   * needsInput() or finished() returns true, to determine why no 
   * further output is produced.
   * @param buffer the output buffer.
   * @param off the offset into buffer where the output should start.
   * @param len the maximum length of the output.
   * @return the number of bytes written to the buffer, 0 if no further
   * output can be produced.  
   * @exception DataFormatException if deflated stream is invalid.
   * @exception IllegalArgumentException if len is lt;eq; 0.
   * @exception IndexOutOfBoundsException if the off and/or len are wrong.
   */
  public static int inflate(java.util.zip.Inflater dis, byte[] buf, int off, int len) throws java.util.zip.DataFormatException
  {
    if (len <= 0)
      throw new IllegalArgumentException("len <= 0");
    int count = 0;
    int more;
    OutputWindow outputWindow = (OutputWindow)Reflection.getfield_A(dis, _outputWindow);
    int mode;
    do
      {
	mode = Reflection.getfield_I(dis, _mode);
	if (mode != DECODE_CHKSUM)
	  {
	    /* Don't give away any output, if we are waiting for the
	     * checksum in the input stream.
	     *
	     * With this trick we have always:
	     *   needsInput() and not finished() 
	     *   implies more output can be produced.  
	     */
	    more = outputWindow.copyOutput(buf, off, len);
	    java.util.zip.Adler32 adler = (java.util.zip.Adler32)Reflection.getfield_A(dis, _adler);
	    adler.update(buf, off, more);
	    off += more;
	    count += more;
	    int totalOut = Reflection.getfield_I(dis, _totalOut);
	    Reflection.putfield_I(dis, _totalOut, totalOut + more);
	    len -= more;
	    if (len == 0)
	      return count;
	  }
	mode = Reflection.getfield_I(dis, _mode);
      }
    while (decode(dis) || (outputWindow.getAvailable() > 0
			&& mode != DECODE_CHKSUM));
    return count;
  }

  /**
   * Returns true, if the input buffer is empty.
   * You should then call setInput(). <br>
   *
   * <em>NOTE</em>: This method also returns true when the stream is finished.
   */
  public static boolean needsInput(java.util.zip.Inflater dis) 
  {
      StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
    return input.needsInput();
  }

  /**
   * Returns true, if a preset dictionary is needed to inflate the input.
   */
  public static boolean needsDictionary(java.util.zip.Inflater dis)
  {
      int mode = Reflection.getfield_I(dis, _mode);
      int neededBits = Reflection.getfield_I(dis, _neededBits);
    return mode == DECODE_DICT && neededBits == 0;
  }

  /**
   * Returns true, if the inflater has finished.  This means, that no
   * input is needed and no output can be produced.
   */
  public static boolean finished(java.util.zip.Inflater dis) 
  {
      int mode = Reflection.getfield_I(dis, _mode);
      OutputWindow outputWindow = (OutputWindow)Reflection.getfield_A(dis, _outputWindow);
    return mode == FINISHED && outputWindow.getAvailable() == 0;
  }

  /**
   * Gets the adler checksum.  This is either the checksum of all
   * uncompressed bytes returned by inflate(), or if needsDictionary()
   * returns true (and thus no output was yet produced) this is the
   * adler checksum of the expected dictionary.
   * @returns the adler checksum.
   */
  public static int getAdler(java.util.zip.Inflater dis)
  {
    return needsDictionary(dis) ? Reflection.getfield_I(dis, _readAdler) : (int) ((java.util.zip.Adler32)Reflection.getfield_A(dis, _adler)).getValue();
  }

  /**
   * Gets the total number of output bytes returned by inflate().
   * @return the total number of output bytes.
   */
  public static int getTotalOut(java.util.zip.Inflater dis)
  {
      return Reflection.getfield_I(dis, _totalOut);
  }

  /**
   * Gets the total number of processed compressed input bytes.
   * @return the total number of bytes of processed input bytes.
   */
  public static int getTotalIn(java.util.zip.Inflater dis)
  {
    return Reflection.getfield_I(dis, _totalIn) - getRemaining(dis);
  }

  /**
   * Gets the number of unprocessed input.  Useful, if the end of the
   * stream is reached and you want to further process the bytes after
   * the deflate stream.  
   * @return the number of bytes of the input which were not processed.
   */
  public static int getRemaining(java.util.zip.Inflater dis)
  {
      StreamManipulator input = (StreamManipulator)Reflection.getfield_A(dis, _input);
    return input.getAvailableBytes();
  }

  /**
   * Frees all objects allocated by the inflater.  There's no reason
   * to call this, since you can just rely on garbage collection (even
   * for the Sun implementation).  Exists only for compatibility
   * with Sun's JDK, where the compressor allocates native memory.
   * If you call any method (even reset) afterwards the behaviour is
   * <i>undefined</i>.  
   * @deprecated Just clear all references to inflater instead.
   */
  public static void end(java.util.zip.Inflater dis)
  {
      Reflection.putfield_A(dis, _outputWindow, null);
      Reflection.putfield_A(dis, _input, null);
      Reflection.putfield_A(dis, _dynHeader, null);
      Reflection.putfield_A(dis, _litlenTree, null);
      Reflection.putfield_A(dis, _distTree, null);
      Reflection.putfield_A(dis, _adler, null);
  }

  /**
   * Finalizes this object.
   */
  protected static void finalize(java.util.zip.Inflater dis)
  {
    /* Exists only for compatibility */
  }

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/Inflater;");
    public static final jq_InstanceField _mode = _class.getOrCreateInstanceField("mode", "I");
    public static final jq_InstanceField _readAdler = _class.getOrCreateInstanceField("readAdler", "I");
    public static final jq_InstanceField _neededBits = _class.getOrCreateInstanceField("neededBits", "I");
    public static final jq_InstanceField _repLength = _class.getOrCreateInstanceField("repLength", "I");
    public static final jq_InstanceField _repDist = _class.getOrCreateInstanceField("repDist", "I");
    public static final jq_InstanceField _uncomprLen = _class.getOrCreateInstanceField("uncomprLen", "I");
    public static final jq_InstanceField _isLastBlock = _class.getOrCreateInstanceField("isLastBlock", "Z");
    public static final jq_InstanceField _totalOut = _class.getOrCreateInstanceField("totalOut", "I");
    public static final jq_InstanceField _totalIn = _class.getOrCreateInstanceField("totalIn", "I");
    public static final jq_InstanceField _nowrap = _class.getOrCreateInstanceField("nowrap", "Z");
    public static final jq_InstanceField _input = _class.getOrCreateInstanceField("input", "LClassLib/sun14_win32/java/util/zip/StreamManipulator;");
    public static final jq_InstanceField _outputWindow = _class.getOrCreateInstanceField("outputWindow", "LClassLib/sun14_win32/java/util/zip/OutputWindow;");
    public static final jq_InstanceField _dynHeader = _class.getOrCreateInstanceField("dynHeader", "LClassLib/sun14_win32/java/util/zip/InflaterDynHeader;");
    public static final jq_InstanceField _litlenTree = _class.getOrCreateInstanceField("litlenTree", "LClassLib/sun14_win32/java/util/zip/InflaterHuffmanTree;");
    public static final jq_InstanceField _distTree = _class.getOrCreateInstanceField("distTree", "LClassLib/sun14_win32/java/util/zip/InflaterHuffmanTree;");
    public static final jq_InstanceField _adler = _class.getOrCreateInstanceField("adler", "Ljava/util/zip/Adler32;");

}
