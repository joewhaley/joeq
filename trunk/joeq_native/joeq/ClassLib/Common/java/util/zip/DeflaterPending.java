/*
 * DeflaterPending.java
 *
 * Created on July 8, 2002, 1:43 AM
 */

package ClassLib.Common.java.util.zip;

/**
 *
 * @author  SUIF User
 * @version 
 */
class DeflaterPending extends PendingBuffer
{
  public DeflaterPending()
  {
    super(DeflaterConstants.PENDING_BUF_SIZE);
  }
}
