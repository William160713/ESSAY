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
    <title>�����D��</title>
</head>
<body>
    <?php echo "<h1 >�����D�� " . $_SESSION['username'] . "</h1>"; ?>
    <!--
    <div style="text-align:right;"><a href="logout.php">�n�X</a>
    -->
    <div style="text-align:right;"><input type="button" value="�n�X"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='logout.php'"></div>
    <div style="text-align:center;">

    <div style="text-align:center;">
    <input type="date"
       value="2021-09-20"
       min="2010-01-01"
       max="2022-12-31"
       step="5">
       </div>
       <br>
       <br>
       <br>
    <div style="text-align:center;"><input type="button" value="�U�@�B"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='Result.php'"></div>
                         

</body>
</html>