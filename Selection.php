<?php 
header("Content-Type:text/html; charset=big5");
session_start();

if (!isset($_SESSION['username'])) {
    header("Location: index.php");
}


?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>�D��</title>
</head>
<body>
    <?php echo "<h1 >�D�� " . $_SESSION['username'] . "</h1>"; ?>
    <!--
    <div style="text-align:right;"><a href="logout.php">�n�X</a>
    -->
    
    <div style="text-align:right;"><input type="button" value="�n�X"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='logout.php'"></div>
    <div style="text-align:center;">
    <div style="text-align:center;">
        



        <br>
        <br>
        <br>
        <h1>��ܧA�A�X�����զX:</h1>
        

      <form action="Rate_of_return.php" method="post">
        <table border="1" width="800"  align="center">
        <tr>
            <td>�O�������</td>
            <td>���ū����</td>
            <td>���������</td>
        </tr>
        <tr>
            <td><input type="radio" name="stock" value="0050">0050</td>
            <td><input type="radio" name="stock" value="2330">2330</td>
            <td><input type="radio" name="stock" value="2615">2615</td>
        </tr>
        

          </table>    

    </div>

    

    <br>
    <br>
    <br>
    
�@
�@  
    



    <div style="text-align:center;"><input type="submit" value="�U�@�B"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='Rate_of_return.php'"></div>
                         
    </form>
</body>
</html>